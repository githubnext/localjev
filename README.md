# LocalJev

A local, Jev-compatible `POST /v1/systemone` API written in TypeScript for
[Bun](https://bun.sh/), backed by DiffusionGemma through an OpenAI-compatible
Chat Completions endpoint.

An optional **Windows / AMD ROCm backend**, tested on Radeon 8060S (gfx1151),
runs the original DiffusionGemma checkpoint in FP16 with PyTorch and
Transformers. See [the ROCm guide](docs/gfx1151.md) for setup, the PowerShell
launcher and an end-to-end HTTP smoke check.

The oMLX defaults target:

- inference server: `http://127.0.0.1:8000`
- model: `diffusiongemma-26B-A4B-it-4bit`
- LocalJev API: `http://127.0.0.1:8080`

## Why a bridge is needed

[Jev](https://typesafe.ai/) uses a typed decision API rather than an OpenAI chat API.
[OpenJev](https://github.com/razorback16/openjev) implements the Jev wire protocol and
obtains probabilities with a special one-step DiffusionGemma **structured read**. Its
backend depends on unmerged vLLM request extensions such as
`diffusion_seed_canvas`, `diffusion_read_only`, and requested token logprobs.

The normal oMLX API does not expose those primitives. LocalJev therefore takes the
portable approach:

1. translate `state` and typed Jev questions into a classification prompt;
2. ask DiffusionGemma for a JSON probability scalar/vector;
3. validate the complete result and retry malformed output;
4. normalize vectors and calculate Jev-compatible choices, expected scores, and
   entropy-based confidence;
5. return the normal Jev response shape.

This is wire-compatible, but not mathematically equivalent to OpenJev's logit read.
The probabilities are generated/self-reported by the model rather than read directly
from its logits. Evaluate their calibration on your own workload before relying on
them for consequential decisions.

## Run with oMLX

Requires Bun 1.4.2+ (for the checked-in lockfile) and a running oMLX server.

```sh
bun install
cp .env.example .env
$EDITOR .env # replace the upstream API-key placeholder
bun run start
```

Bun loads `.env` automatically. Alternatively, set the key in your shell before
starting the server:

```fish
# fish
set -gx LOCALJEV_UPSTREAM_API_KEY 'your-local-omlx-key'
```

```sh
# bash/zsh
export LOCALJEV_UPSTREAM_API_KEY='your-local-omlx-key'
```

LocalJev listens on `http://127.0.0.1:8080`. Check that the configured model is
available:

```bash
curl http://127.0.0.1:8080/ready
```

Make a decision:

```bash
curl http://127.0.0.1:8080/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "Hi, I have been trying to connect Stripe but keep getting a 403 error.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payment or subscription issues",
          "technical": "Bugs or integration problems",
          "sales": "Pricing or account questions"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated does the customer appear?",
        "criteria": ["Calm", "Frustrated but civil", "Very angry"]
      },
      "urgent": {
        "type": "noul",
        "instructions": "Does this require an immediate response?"
      }
    }
  }'
```

## Use the TypeSafe SDK

The SDK requires an API-key value. LocalJev accepts any value unless
`LOCALJEV_API_KEY` is configured. Set the SDK environment for your shell:

```fish
# fish
set -gx TYPESAFE_BASE_URL http://127.0.0.1:8080
set -gx TYPESAFE_API_KEY local
```

```sh
# bash/zsh
export TYPESAFE_BASE_URL=http://127.0.0.1:8080
export TYPESAFE_API_KEY=local
```

```python
from typesafe_sdk import TypeSafeClient

client = TypeSafeClient()
response = client.system_one(
    "I was charged twice this month.",
    {
        "billing": {
            "type": "noul",
            "instructions": "Is this a billing issue?",
        }
    },
)
print(response.nouls["billing"].noul)
```

`jev-latest` and `jev-preview` are accepted aliases so SDK defaults work unchanged.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LOCALJEV_UPSTREAM` | `http://127.0.0.1:8000` | OpenAI-compatible base URL, with or without `/v1` |
| `LOCALJEV_UPSTREAM_API_KEY` | empty | Bearer key sent to the inference server |
| `LOCALJEV_UPSTREAM_MODEL` | `diffusiongemma-26B-A4B-it-4bit` | Upstream model identifier |
| `LOCALJEV_API_KEY` | empty | Optional Bearer key required from LocalJev clients |
| `LOCALJEV_HOST` | `127.0.0.1` | Listen address |
| `LOCALJEV_PORT` | `8080` | Listen port |
| `LOCALJEV_TIMEOUT` | `180` | Upstream timeout in seconds |
| `LOCALJEV_MAX_INFLIGHT` | `2` | Concurrent calls admitted upstream |
| `LOCALJEV_MAX_QUEUE` | `64` | Waiting decisions before HTTP 529 |
| `LOCALJEV_MALFORMED_RETRIES` | `2` | Corrective retries for invalid model JSON |
| `LOCALJEV_MAX_OUTPUT_TOKENS` | `2048` | Per-completion output ceiling |
| `LOCALJEV_QUESTIONS_PER_CALL` | `16` | Chunking limit per model call |
| `LOCALJEV_OUTCOMES_PER_CALL` | `128` | Choice/score outcomes per model call |

Bun automatically loads `.env`, so you can also copy `.env.example`, replace its
placeholder, and run the server.

## Development

```bash
bun install
bun test
bun run typecheck
bun run smoke       # live call to the configured inference server
bun run smoke:http  # /ready and all three decision types through LocalJev HTTP
```

## Evaluate different models

The repeatable bake-off uses public gold labels for news categorization (AG News),
yes/no reading comprehension (BoolQ), and five-level sentiment (SST-5). It runs the
same LocalJev engine against five installed models, comparing quality, calibration,
retries, and full-decision latency at two actual input lengths.

```sh
# Quick integration check (30 requests, not a meaningful quality sample)
bun run eval --out eval/runs/pilot --limit 3

# 5 models × 120 labeled examples × 2 input lengths = 1,200 requests
bun run eval --out eval/runs/my-bakeoff

# Regenerate a completed or partial report without running inference
bun run eval:report eval/runs/my-bakeoff
```

Requires oMLX and the upstream key in `.env`; no running LocalJev HTTP server or
Python is needed. See [the evaluation guide](docs/evaluation.md) for pinned data
sources, methodology, configuration, resuming runs, and limitations.

The [first completed bake-off](docs/evaluation-results-2026-09-18.md) includes
1,200 requests on an M5 Max. Gemma 4 26B-A4B and Qwen3.6 were the strongest overall
candidates in this small screening sample; the report includes per-task results,
latency, context effects, and caveats rather than claiming a definitive winner.

## Should you use LM Studio instead?

Not currently for this model. As of September 18, 2026, DiffusionGemma support is
still tracked as open in both
[`lmstudio-ai/mlx-engine#336`](https://github.com/lmstudio-ai/mlx-engine/issues/336)
and
[`lmstudio-ai/lmstudio-bug-tracker#2037`](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2037).
The reported MLX backend fails to load `diffusion_gemma`, while the normal llama.cpp
backend reports an unknown architecture. oMLX already loads and serves your exact
checkpoint successfully in the Mac setup. For Windows gfx1151, see the
[optional ROCm backend](docs/gfx1151.md).

Even after LM Studio adds ordinary generation support, changing runners alone will
not make the result OpenJev-equivalent. The runner must expose seeded diffusion
canvases, read-only denoising, and selected-token logits/logprobs. If LM Studio only
provides standard Chat Completions, LocalJev can use it by changing
`LOCALJEV_UPSTREAM`, but the probability path remains prompted/self-reported.

For direct model probabilities, the best paths are:

1. add the structured-read primitives to oMLX's DiffusionGemma lane and consume them
   here; or
2. run OpenJev's patched vLLM backend on a supported NVIDIA machine.
