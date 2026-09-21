"""CPU-only HTTP/queue tests; never import torch or download model weights."""

import copy
import http.client
import json
from pathlib import Path
import select
import sys
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rocm_server import (  # noqa: E402
    Generation, InferenceService, Limits, RequestError,
    final_text, make_server, parse_request, trim_generated_tokens,
)


class FakeRuntime:
    model_id = "test-model"
    info = {"architecture": "gfx1151", "device": "cuda:0"}

    def __init__(self):
        self.requests = []
        self.failure = False

    def generate(self, request):
        self.requests.append(request)
        if self.failure:
            raise RuntimeError("private backend error")
        return Generation('{"billing":0.9}', 21, 8, "stop")


def request_body(**overrides):
    body = {"model": "test-model", "messages": [{"role": "user", "content": "Charged twice"}], "max_tokens": 32, "seed": 1234}
    body.update(overrides)
    return body


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.service = InferenceService(self.runtime, Limits(max_body_bytes=4096))
        self.server = make_server(self.service, "127.0.0.1", 0, "test-key")
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def call(self, method, path, payload=None, key="test-key", content_type="application/json"):
        connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
        headers = {"Authorization": f"Bearer {key}", "Content-Type": content_type}
        data = payload if isinstance(payload, (str, bytes)) else json.dumps(payload) if payload is not None else None
        connection.request(method, path, body=data, headers=headers)
        response = connection.getresponse()
        status = response.status
        result = json.loads(response.read())
        connection.close()
        return status, result

    def test_health_and_model_inventory(self):
        status, health = self.call("GET", "/health")
        self.assertEqual(status, 200)
        self.assertEqual(health["runtime"]["architecture"], "gfx1151")
        self.assertEqual(health["structured_output"], "prompt_only")
        status, models = self.call("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(models["data"][0]["id"], "test-model")

    def test_localjev_request_and_openai_response(self):
        schema = {"type": "object", "properties": {"billing": {"type": "number"}}, "required": ["billing"]}
        body = request_body(temperature=0.0, chat_template_kwargs={"enable_thinking": False}, response_format={"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": schema}})
        original = copy.deepcopy(body)
        status, response = self.call("POST", "/v1/chat/completions", body)
        self.assertEqual(status, 200)
        self.assertEqual(response["model"], "test-model")
        self.assertEqual(response["choices"][0]["message"]["content"], '{"billing":0.9}')
        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(response["usage"], {"prompt_tokens": 21, "completion_tokens": 8, "total_tokens": 29})
        self.assertEqual(self.runtime.requests[0].seed, 1234)
        self.assertIn(json.dumps(schema), self.runtime.requests[0].messages[0]["content"])
        self.assertEqual(body, original)

    def test_authentication_and_model_mismatch_do_not_generate(self):
        self.assertEqual(self.call("GET", "/health", key="wrong")[0], 401)
        self.assertEqual(self.call("POST", "/v1/chat/completions", request_body(model="other"))[0], 404)
        self.assertEqual(self.runtime.requests, [])

    def test_invalid_transport_and_json_do_not_generate(self):
        cases = [("not-json", "application/json", 400), ('{"x":NaN}', "application/json", 400), ("x" * 4097, "application/json", 413), ("{}", "text/plain", 415)]
        for payload, content_type, expected in cases:
            with self.subTest(expected=expected, payload=payload[:20]):
                self.assertEqual(self.call("POST", "/v1/chat/completions", payload, content_type=content_type)[0], expected)
        self.assertEqual(self.runtime.requests, [])

    def test_media_type_rejection_consumes_split_body_without_connection_reset(self):
        for attempt in range(20):
            with self.subTest(attempt=attempt):
                connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)
                try:
                    connection.putrequest("POST", "/v1/chat/completions")
                    connection.putheader("Authorization", "Bearer test-key")
                    connection.putheader("Content-Type", "text/plain")
                    connection.putheader("Content-Length", "2")
                    connection.endheaders(b"{")
                    # The old handler closed here without consuming the body;
                    # sending the trailing byte could reset its error response.
                    readable, _, _ = select.select([connection.sock], [], [], 0.01)
                    self.assertEqual(readable, [], "Server rejected before reading the complete bounded body")
                    connection.send(b"}")
                    response = connection.getresponse()
                    self.assertEqual(response.status, 415)
                    self.assertIn("Content-Type", json.loads(response.read())["error"]["message"])
                finally:
                    connection.close()
        self.assertEqual(self.runtime.requests, [])

    def test_backend_errors_are_redacted_and_capacity_is_released(self):
        self.runtime.failure = True
        with self.assertLogs("localjev.rocm", level="ERROR"):
            status, response = self.call("POST", "/v1/chat/completions", request_body())
        self.assertEqual(status, 500)
        self.assertNotIn("private backend error", json.dumps(response))
        self.runtime.failure = False
        self.assertEqual(self.call("POST", "/v1/chat/completions", request_body())[0], 200)


class ValidationTests(unittest.TestCase):
    def test_rejects_unsupported_or_unbounded_requests(self):
        cases = [
            {"stream": True}, {"n": 2}, {"max_tokens": 0}, {"max_tokens": True},
            {"max_tokens": 4096}, {"seed": -1}, {"temperature": float("nan")},
            {"temperature": True}, {"messages": []}, {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]},
            {"messages": [{"role": "tool", "content": "data"}]}, {"tools": []},
            {"chat_template_kwargs": {"enable_thinking": True}},
            {"response_format": {"type": "json_schema", "json_schema": {"schema": []}}},
        ]
        for overrides in cases:
            with self.subTest(overrides=overrides), self.assertRaises(RequestError):
                parse_request(request_body(**overrides), "test-model", Limits())

    def test_preserves_existing_system_prompt_without_mutating_input(self):
        body = request_body(messages=[{"role": "system", "content": "Classify."}, {"role": "user", "content": "Hi"}], response_format={"type": "json_object"})
        result = parse_request(body, "test-model", Limits())
        self.assertTrue(result.messages[0]["content"].startswith("Classify."))
        self.assertEqual(body["messages"][0]["content"], "Classify.")
        self.assertEqual(len(result.messages), 2)

    def test_diffusion_canvas_respects_output_budget_and_eos(self):
        self.assertEqual(trim_generated_tokens([4, 5, 6, 1, 0, 0], [1], 3), ([4, 5, 6], "length"))
        self.assertEqual(trim_generated_tokens([4, 5, 1, 0, 0], [1], 3), ([4, 5, 1], "stop"))
        self.assertEqual(trim_generated_tokens([4, 1, 0, 0], [1], 10), ([4, 1], "stop"))

    def test_reasoning_markers_do_not_contaminate_probability_json(self):
        self.assertEqual(final_text('<think>analysis</think>\n{"p": 0.9}'), '{"p": 0.9}')
        self.assertEqual(final_text('<|channel>thought\n<channel|>{"p": 0.9}'), '{"p": 0.9}')
        self.assertEqual(final_text('{"p": 0.9}'), '{"p": 0.9}')

    def test_full_admission_queue_rejects_without_concurrent_generation(self):
        runtime = FakeRuntime()
        started = threading.Event()
        release = threading.Event()
        original_generate = runtime.generate

        def blocked_generate(request):
            started.set()
            if not release.wait(timeout=3):
                raise RuntimeError("test timed out")
            return original_generate(request)

        runtime.generate = blocked_generate
        service = InferenceService(runtime, Limits(max_queue=0))
        worker = threading.Thread(target=lambda: service.complete(request_body()))
        worker.start()
        try:
            self.assertTrue(started.wait(timeout=2))
            with self.assertRaises(RequestError) as captured:
                service.complete(request_body())
            self.assertEqual(captured.exception.status, 429)
        finally:
            release.set()
            worker.join(timeout=3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(runtime.requests), 1)
        service.complete(request_body())
        self.assertEqual(len(runtime.requests), 2)

    def test_queue_timeout_releases_admission(self):
        service = InferenceService(FakeRuntime(), Limits(max_queue=0, queue_timeout=0.01))
        service.inference.acquire()
        try:
            with self.assertRaises(RequestError) as captured:
                service.complete(request_body())
            self.assertEqual(captured.exception.status, 429)
        finally:
            service.inference.release()
        self.assertEqual(service.complete(request_body())["object"], "chat.completion")


if __name__ == "__main__":
    unittest.main()
