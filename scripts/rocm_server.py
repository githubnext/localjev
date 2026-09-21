"""Local text-only Chat Completions adapter for ROCm; JSON schemas are prompted.

Install requirements-rocm.txt into an environment with a working HIP PyTorch.
The default model is the original DiffusionGemma checkpoint, not an MLX quant.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
import re
import threading
import time
import uuid
from typing import Any


DEFAULT_MODEL = "google/diffusiongemma-26B-A4B-it"
LOG = logging.getLogger("localjev.rocm")


class RequestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Limits:
    max_body_bytes: int = 1_048_576
    max_messages: int = 64
    max_input_tokens: int = 8192
    max_output_tokens: int = 2048
    max_queue: int = 2
    queue_timeout: float = 180.0


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict[str, str]]
    max_tokens: int
    temperature: float
    seed: int | None


@dataclass(frozen=True)
class Generation:
    content: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


def integer(value: Any, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise RequestError(f"{name} must be an integer from {minimum} to {maximum}")
    return value


def parse_request(body: Any, model_id: str, limits: Limits) -> ChatRequest:
    if not isinstance(body, dict):
        raise RequestError("The request must be a JSON object")
    if body.get("model") != model_id:
        raise RequestError(f"Only model {model_id!r} is loaded", 404)
    if body.get("stream", False) is not False:
        raise RequestError("Only non-streaming requests are supported")
    integer(body.get("n", 1), "n", 1, 1)
    for field in ("tools", "tool_choice", "stop", "logprobs", "top_logprobs"):
        if body.get(field) is not None:
            raise RequestError(f"{field} is not supported by this text adapter")
    messages = body.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= limits.max_messages:
        raise RequestError(f"messages must contain 1 to {limits.max_messages} messages")
    clean = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant"):
            raise RequestError("Each message needs a system, user, or assistant role")
        content = message.get("content")
        if not isinstance(content, str):
            raise RequestError("Only string message content is supported; no image/audio inputs")
        clean.append({"role": message["role"], "content": content})
    max_tokens = integer(body.get("max_tokens", min(512, limits.max_output_tokens)), "max_tokens", 1, limits.max_output_tokens)
    if "max_completion_tokens" in body:
        raise RequestError("Use max_tokens instead of max_completion_tokens")
    temperature = body.get("temperature", 0.0)
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise RequestError("temperature must be a finite number from 0 to 2")
    seed = body.get("seed")
    if seed is not None:
        seed = integer(seed, "seed", 0, 2**32 - 1)
    template = body.get("chat_template_kwargs", {})
    if not isinstance(template, dict) or set(template) - {"enable_thinking"}:
        raise RequestError("Only chat_template_kwargs.enable_thinking=false is supported")
    if template.get("enable_thinking", False) is not False:
        raise RequestError("Thinking must be disabled for this decision endpoint")
    response_format = body.get("response_format")
    if response_format is not None:
        if not isinstance(response_format, dict):
            raise RequestError("response_format must be an object")
        format_type = response_format.get("type")
        instruction = None
        if format_type == "json_schema":
            definition = response_format.get("json_schema")
            if not isinstance(definition, dict) or not isinstance(definition.get("schema"), dict):
                raise RequestError("response_format.json_schema.schema must be an object")
            instruction = "Return only JSON matching this schema, with no explanation or markdown:\n" + json.dumps(definition["schema"], ensure_ascii=False)
        elif format_type == "json_object":
            instruction = "Return only a JSON object, with no explanation or markdown."
        elif format_type != "text":
            raise RequestError("response_format.type must be text, json_object, or json_schema")
        # This is a prompt instruction, not constrained decoding or schema enforcement.
        if instruction:
            if clean[0]["role"] == "system":
                clean[0]["content"] += "\n\n" + instruction
            else:
                clean.insert(0, {"role": "system", "content": instruction})
    return ChatRequest(clean, max_tokens, float(temperature), seed)


def final_text(text: str) -> str:
    """Remove complete reasoning blocks while preserving the final answer."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel>thought\s*.*?<channel\|>", "", text, flags=re.DOTALL)
    return text.strip()


def trim_generated_tokens(tokens: list[int], eos_ids: list[int], maximum: int) -> tuple[list[int], str]:
    # Diffusion generates a whole canvas, which can extend past max_tokens.
    for index, token in enumerate(tokens[:maximum]):
        if token in eos_ids:
            return tokens[:index + 1], "stop"
    return tokens[:maximum], "length" if len(tokens) >= maximum else "stop"


class TransformersRuntime:
    def __init__(self, args: argparse.Namespace, limits: Limits):
        # Import lazily so request validation tests never load torch or touch a GPU.
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer, DiffusionGemmaForBlockDiffusion

        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("A working ROCm/HIP PyTorch GPU is required; CPU/CUDA fallback is disabled")
        if not 0 <= args.device < torch.cuda.device_count():
            raise RuntimeError(f"ROCm device {args.device} is not available")
        properties = torch.cuda.get_device_properties(args.device)
        architecture = getattr(properties, "gcnArchName", "unknown")
        if architecture.split(":", 1)[0] != "gfx1151" and not args.allow_other_gpu:
            raise RuntimeError(f"Expected gfx1151, found {architecture}; use --allow-other-gpu for another HIP GPU")
        self.torch = torch
        self.device = torch.device(f"cuda:{args.device}")
        torch.cuda.set_device(self.device)
        self.model_id = args.model
        self.limits = limits
        self.info = {
            "device": str(self.device), "gpu": properties.name, "architecture": architecture,
            "hip": torch.version.hip, "torch": torch.__version__, "transformers": transformers.__version__,
            "dtype": args.dtype, "attention": "sdpa", "experts": "eager",
        }
        LOG.info("ROCm runtime: %s", json.dumps(self.info))
        load_options = {"local_files_only": args.local_files_only, "trust_remote_code": False}
        if args.revision:
            load_options["revision"] = args.revision
        config = AutoConfig.from_pretrained(args.model, **load_options)
        self.is_diffusion = config.model_type == "diffusion_gemma"
        if self.is_diffusion:
            model_class = DiffusionGemmaForBlockDiffusion
            self.processor = AutoProcessor.from_pretrained(args.model, **load_options)
            self.tokenizer = self.processor.tokenizer
        else:
            model_class = AutoModelForCausalLM
            self.processor = AutoTokenizer.from_pretrained(args.model, **load_options)
            self.tokenizer = self.processor
        self.info["model_type"] = config.model_type
        self.model = model_class.from_pretrained(
            args.model, config=config, dtype=getattr(torch, args.dtype),
            device_map={"": str(self.device)}, attn_implementation="sdpa",
            # The 5.11 grouped_mm auto-selection checks CUDA-shaped capability
            # without distinguishing HIP. Keep MoE on ordinary PyTorch linears.
            experts_implementation="eager", **load_options,
        ).eval()
        misplaced = [name for name, value in self.model.named_parameters() if value.device != self.device]
        if misplaced:
            raise RuntimeError(f"Model was not fully placed on {self.device}: {misplaced[:3]}")
        torch.cuda.synchronize(self.device)
        LOG.info("Loaded %s (%s) entirely on %s", args.model, config.model_type, architecture)
        if self.is_diffusion:
            LOG.info("Diffusion sampling uses checkpoint t_min/t_max; request temperature does not replace its schedule. Dynamic cache disables compilation.")

    def generate(self, request: ChatRequest) -> Generation:
        inputs = self.processor.apply_chat_template(
            request.messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        )
        prompt_tokens = inputs["input_ids"].shape[-1]
        if prompt_tokens > self.limits.max_input_tokens:
            raise RequestError(f"Prompt has {prompt_tokens} tokens; maximum is {self.limits.max_input_tokens}")
        inputs = inputs.to(self.device)
        # Transformers 5.11 DiffusionGemma always returns .sequences. Its
        # compile path is entered only for a static (compileable) cache.
        options = {"max_new_tokens": request.max_tokens, "cache_implementation": "dynamic"}
        if not self.is_diffusion:
            options["do_sample"] = request.temperature > 0
            options["disable_compile"] = True
            if request.temperature > 0:
                options["temperature"] = request.temperature
            if self.tokenizer.pad_token_id is None:
                options["pad_token_id"] = self.tokenizer.eos_token_id
        if request.seed is not None:
            self.torch.manual_seed(request.seed)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        sequences = output.sequences if hasattr(output, "sequences") else output
        generated = sequences[0, prompt_tokens:].tolist()
        eos_ids = self.model.generation_config.eos_token_id
        if eos_ids is None:
            eos_ids = self.tokenizer.eos_token_id
        if not isinstance(eos_ids, list):
            eos_ids = [] if eos_ids is None else [eos_ids]
        generated, finish_reason = trim_generated_tokens(generated, eos_ids, request.max_tokens)
        # Skip control tokens individually after removing thought-channel text.
        text = final_text(self.tokenizer.decode(generated, skip_special_tokens=False))
        for token in self.tokenizer.all_special_tokens:
            text = text.replace(token, "")
        return Generation(text.strip(), prompt_tokens, len(generated), finish_reason)


class InferenceService:
    def __init__(self, runtime: Any, limits: Limits):
        self.runtime = runtime
        self.limits = limits
        self.admission = threading.BoundedSemaphore(1 + limits.max_queue)
        self.inference = threading.Lock()

    def complete(self, body: Any) -> dict[str, Any]:
        request = parse_request(body, self.runtime.model_id, self.limits)
        if not self.admission.acquire(blocking=False):
            raise RequestError("Inference queue is full; retry shortly", 429)
        try:
            if not self.inference.acquire(timeout=self.limits.queue_timeout):
                raise RequestError("Timed out waiting for inference; retry shortly", 429)
            try:
                started = time.monotonic()
                result = self.runtime.generate(request)
                LOG.info("Generated %d tokens in %.2f seconds (%s)", result.completion_tokens, time.monotonic() - started, result.finish_reason)
            finally:
                self.inference.release()
        finally:
            self.admission.release()
        return {
            "id": "chatcmpl-" + uuid.uuid4().hex, "object": "chat.completion",
            "created": int(time.time()), "model": self.runtime.model_id,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": result.content}, "finish_reason": result.finish_reason}],
            "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens, "total_tokens": result.prompt_tokens + result.completion_tokens},
        }


def make_server(service: InferenceService, host: str, port: int, api_key: str = "") -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, format: str, *args):
            LOG.info(format, *args)

        def send_json(self, status: int, body: dict[str, Any]):
            encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            if status == 429:
                self.send_header("Retry-After", "1")
            self.end_headers()
            self.wfile.write(encoded)

        def authorize(self):
            supplied = self.headers.get("Authorization", "")
            if api_key and not hmac.compare_digest(supplied.encode(), ("Bearer " + api_key).encode()):
                raise RequestError("Invalid upstream API key", 401)

        def run_request(self, action):
            try:
                self.authorize()
                action()
            except RequestError as error:
                self.send_json(error.status, {"error": {"message": str(error), "type": "invalid_request_error" if error.status < 500 else "server_error"}})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                LOG.warning("Client disconnected or request body timed out")
            except Exception:
                LOG.exception("Inference request failed")
                self.send_json(500, {"error": {"message": "Inference failed; inspect the backend log", "type": "server_error"}})

        def do_GET(self):
            def respond():
                if self.path == "/health":
                    self.send_json(200, {"status": "ok", "model": service.runtime.model_id, "runtime": service.runtime.info, "structured_output": "prompt_only"})
                elif self.path == "/v1/models":
                    self.send_json(200, {"object": "list", "data": [{"id": service.runtime.model_id, "object": "model", "created": 0, "owned_by": "local"}]})
                else:
                    raise RequestError("Unknown endpoint", 404)
            self.run_request(respond)

        def do_POST(self):
            def respond():
                if self.path != "/v1/chat/completions":
                    raise RequestError("Unknown endpoint", 404)
                if self.headers.get("Transfer-Encoding"):
                    raise RequestError("Chunked request bodies are not supported")
                try:
                    length = int(self.headers.get("Content-Length", ""))
                except ValueError:
                    raise RequestError("Content-Length is required", 411) from None
                if not 0 < length <= service.limits.max_body_bytes:
                    raise RequestError("Request body exceeds the configured size limit", 413)
                payload = self.rfile.read(length)
                if len(payload) != length:
                    raise RequestError("Incomplete request body")
                # Consume a bounded body before rejecting its media type. Closing
                # a Windows socket with unread bytes can reset the connection
                # before the client receives the HTTP error response.
                if self.headers.get_content_type() != "application/json":
                    raise RequestError("Content-Type must be application/json", 415)
                def reject_constant(value):
                    raise ValueError(value)
                try:
                    body = json.loads(payload, parse_constant=reject_constant)
                except (ValueError, UnicodeDecodeError, RecursionError):
                    raise RequestError("Request body must contain valid JSON") from None
                self.send_json(200, service.complete(body))
            self.run_request(respond)

    return ThreadingHTTPServer((host, port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--allow-other-gpu", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--max-input-tokens", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--max-queue", type=int, default=2)
    parser.add_argument("--queue-timeout", type=float, default=180.0)
    args = parser.parse_args()
    if args.max_input_tokens < 1 or args.max_output_tokens < 1 or args.max_queue < 0 or not math.isfinite(args.queue_timeout) or args.queue_timeout <= 0:
        parser.error("Token limits and queue timeout must be positive; max-queue must be nonnegative")
    if not 1 <= args.port <= 65535:
        parser.error("port must be from 1 to 65535")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    limits = Limits(max_input_tokens=args.max_input_tokens, max_output_tokens=args.max_output_tokens, max_queue=args.max_queue, queue_timeout=args.queue_timeout)
    runtime = TransformersRuntime(args, limits)
    service = InferenceService(runtime, limits)
    server = make_server(service, args.host, args.port, os.environ.get("LOCALJEV_UPSTREAM_API_KEY", ""))
    LOG.info("Ready at http://%s:%d; schema output is prompted, not grammar constrained", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOG.info("Stopping ROCm backend")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
