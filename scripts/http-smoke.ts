import type { SystemOneRequest } from "../src/types";

// The API derives its seed from state and questions, so keep this request fixed.
const request: SystemOneRequest = {
  model: "localjev-latest",
  state: "Hi, I have been trying to connect Stripe but keep getting a 403 error.",
  questions: {
    department: {
      type: "choice",
      instructions: "Which team should handle this?",
      criteria: {
        billing: "Payment or subscription issues",
        technical: "Bugs or integration problems",
        sales: "Pricing or account questions",
      },
    },
    frustration: {
      type: "score",
      instructions: "How frustrated does the customer appear?",
      criteria: ["Calm", "Frustrated but civil", "Very angry"],
    },
    urgent: {
      type: "noul",
      instructions: "Does this require an immediate response?",
      criteria: null,
    },
  },
};

function object(value: unknown, path: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${path} must be an object`);
  }
  return value as Record<string, unknown>;
}

function keys(value: Record<string, unknown>, expected: string[], path: string): void {
  if (
    Object.keys(value).length !== expected.length ||
    expected.some((key) => !Object.hasOwn(value, key))
  ) {
    throw new Error(`${path} must contain exactly: ${expected.join(", ")}`);
  }
}

function boundedNumber(value: unknown, maximum: number, path: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0 || value > maximum) {
    throw new Error(`${path} must be a finite number between 0 and ${maximum}`);
  }
  return value;
}

function distribution(value: unknown, labels: string[], path: string): void {
  const probabilities = object(value, path);
  keys(probabilities, labels, path);
  const total = labels.reduce(
    (sum, label) => sum + boundedNumber(probabilities[label], 1, `${path}.${label}`),
    0,
  );
  if (Math.abs(total - 1) > 1e-6) {
    throw new Error(`${path} must sum to 1 (received ${total})`);
  }
}

function validateResponse(raw: unknown): Record<string, unknown> {
  const body = object(raw, "response");
  if (typeof body.model !== "string" || body.model.length === 0) {
    throw new Error("response.model must be a nonempty string");
  }
  const answers = object(body.answers, "answers");
  keys(answers, Object.keys(request.questions), "answers");
  for (const [name, question] of Object.entries(request.questions)) {
    const path = `answers.${name}`;
    const answer = object(answers[name], path);
    if (answer.type !== question.type) throw new Error(`${path}.type must be ${question.type}`);
    if (question.type === "noul") {
      boundedNumber(answer.noul, 1, `${path}.noul`);
      continue;
    }
    boundedNumber(answer.confidence, 1, `${path}.confidence`);
    if (question.type === "choice") {
      const labels = Object.keys(question.criteria);
      distribution(answer.probabilities, labels, `${path}.probabilities`);
      if (typeof answer.choice !== "string" || !labels.includes(answer.choice)) {
        throw new Error(`${path}.choice must be one of the requested labels`);
      }
    } else {
      const labels = question.criteria.map((_, index) => String(index));
      distribution(answer.probabilities, labels, `${path}.probabilities`);
      boundedNumber(answer.score, question.criteria.length - 1, `${path}.score`);
      const legend = object(answer.legend, `${path}.legend`);
      keys(legend, labels, `${path}.legend`);
      for (const [index, criterion] of question.criteria.entries()) {
        if (legend[String(index)] !== criterion) {
          throw new Error(`${path}.legend.${index} must match the requested criterion`);
        }
      }
    }
  }
  const usage = object(body.usage, "usage");
  for (const name of ["input_tokens", "output_tokens"]) {
    const count = boundedNumber(usage[name], Number.MAX_SAFE_INTEGER, `usage.${name}`);
    if (!Number.isInteger(count)) throw new Error(`usage.${name} must be an integer`);
  }
  return body;
}

export async function runHttpSmoke(options: {
  url: string;
  apiKey?: string;
  timeoutMs?: number;
}): Promise<Record<string, unknown>> {
  const base = new URL(options.url.replace(/\/+$/, "") + "/");
  if (!["http:", "https:"].includes(base.protocol) || base.username || base.password || base.search || base.hash) {
    throw new Error("LocalJev URL must be HTTP(S) without credentials, query parameters, or a fragment");
  }
  const timeoutMs = options.timeoutMs ?? 300_000;
  if (!Number.isSafeInteger(timeoutMs) || timeoutMs <= 0) {
    throw new Error("Smoke timeout must be a positive integer number of milliseconds");
  }
  const headers: HeadersInit = {
    "content-type": "application/json",
    ...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}),
  };
  async function fetchJson(path: string, init?: RequestInit): Promise<unknown> {
    const response = await fetch(new URL(path, base), {
      ...init,
      headers,
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (response.status !== 200) {
      throw new Error(`${path} returned HTTP ${response.status}: ${(await response.text()).slice(0, 1_000)}`);
    }
    try {
      return await response.json();
    } catch {
      throw new Error(`${path} did not return valid JSON`);
    }
  }
  const ready = object(await fetchJson("ready"), "ready");
  if (ready.status !== "ready" || typeof ready.upstream_model !== "string" || !ready.upstream_model) {
    throw new Error("/ready must report status=ready and a nonempty upstream_model");
  }
  const started = performance.now();
  const body = validateResponse(await fetchJson("v1/systemone", {
    method: "POST",
    body: JSON.stringify(request),
  }));
  return {
    status: "ok",
    url: base.href,
    upstream_model: ready.upstream_model,
    latency_ms: Math.round((performance.now() - started) * 100) / 100,
    model: body.model,
    answers: body.answers,
    usage: body.usage,
  };
}

if (import.meta.main) {
  try {
    const host = process.env.LOCALJEV_HOST ?? "127.0.0.1";
    const clientHost = host === "0.0.0.0" ? "127.0.0.1" : host === "::" ? "[::1]" : host;
    const evidence = await runHttpSmoke({
      url: process.env.LOCALJEV_URL ?? `http://${clientHost.includes(":") && !clientHost.startsWith("[") ? `[${clientHost}]` : clientHost}:${process.env.LOCALJEV_PORT ?? "8080"}`,
      apiKey: process.env.LOCALJEV_API_KEY ?? "",
      timeoutMs: Number(process.env.LOCALJEV_SMOKE_TIMEOUT ?? "300") * 1_000,
    });
    console.log(JSON.stringify(evidence));
  } catch (error) {
    console.error(`HTTP smoke failed: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  }
}
