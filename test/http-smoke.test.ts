import { describe, expect, test } from "bun:test";
import { fileURLToPath } from "node:url";

import { runHttpSmoke } from "../scripts/http-smoke";
import { loadSettings } from "../src/config";
import type { DecisionEngine } from "../src/engine";
import { LocalJevApp } from "../src/server";

const validResponse = {
  model: "localjev-0.2",
  answers: {
    department: {
      type: "choice",
      choice: "technical",
      probabilities: { billing: 0.1, technical: 0.8, sales: 0.1 },
      confidence: 0.4,
    },
    frustration: {
      type: "score",
      score: 0.9,
      legend: { "0": "Calm", "1": "Frustrated but civil", "2": "Very angry" },
      probabilities: { "0": 0.2, "1": 0.7, "2": 0.1 },
      confidence: 0.2,
    },
    urgent: { type: "noul", noul: 0.3 },
  },
  usage: { input_tokens: 120, output_tokens: 40 },
};

describe("HTTP smoke", () => {
  test("calls a listening authenticated LocalJev server with all question types and a stable seed", async () => {
    const seeds: number[] = [];
    const engine: DecisionEngine = {
      async ready() { return true; },
      async decide(questions, _state, seed) {
        expect(Object.values(questions).map((question) => question.type)).toEqual(["choice", "score", "noul"]);
        seeds.push(seed);
        return {
          answers: validResponse.answers as Awaited<ReturnType<DecisionEngine["decide"]>>["answers"],
          inputTokens: 120,
          outputTokens: 40,
        };
      },
    };
    const app = new LocalJevApp(loadSettings({ apiKey: "smoke-test-key", upstreamModel: "test-model" }), engine);
    const server = Bun.serve({ hostname: "127.0.0.1", port: 0, fetch: (request) => app.fetch(request) });
    try {
      const options = { url: server.url.href, apiKey: "smoke-test-key" };
      const evidence = await runHttpSmoke(options);
      expect(evidence).toMatchObject({
        status: "ok", upstream_model: "test-model", ...validResponse,
      });
      expect(evidence.latency_ms).toBeGreaterThanOrEqual(0);
      await runHttpSmoke(options);
      expect(seeds).toHaveLength(2);
      expect(seeds[0]).toBe(seeds[1]);
    } finally {
      await server.stop(true);
      await app.close();
    }
  });

  test("fails before inference when readiness is unavailable", async () => {
    const paths: string[] = [];
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      fetch(request) {
        paths.push(new URL(request.url).pathname);
        return Response.json({ status: "unavailable" }, { status: 503 });
      },
    });
    try {
      await expect(runHttpSmoke({ url: server.url.href })).rejects.toThrow("ready returned HTTP 503");
      expect(paths).toEqual(["/ready"]);
    } finally {
      await server.stop(true);
    }
  });

  test.each([
    ["incorrect question type", (body: typeof validResponse) => { body.answers.urgent.type = "score"; }, "answers.urgent.type"],
    ["probability sum", (body: typeof validResponse) => { body.answers.department.probabilities.billing = 0.9; }, "must sum to 1"],
    ["noul bounds", (body: typeof validResponse) => { body.answers.urgent.noul = 1.1; }, "answers.urgent.noul"],
    ["token counts", (body: typeof validResponse) => { body.usage.input_tokens = -1; }, "usage.input_tokens"],
  ] as const)("rejects invalid %s over HTTP", async (_name, mutate, expected) => {
    const body = structuredClone(validResponse);
    mutate(body);
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      fetch(request) {
        return Response.json(new URL(request.url).pathname === "/ready"
          ? { status: "ready", upstream_model: "test-model" }
          : body);
      },
    });
    try {
      await expect(runHttpSmoke({ url: server.url.href })).rejects.toThrow(expected);
    } finally {
      await server.stop(true);
    }
  });

  test("CLI exits nonzero when readiness fails", async () => {
    const server = Bun.serve({
      hostname: "127.0.0.1", port: 0,
      fetch: () => Response.json({ status: "unavailable" }, { status: 503 }),
    });
    try {
      const child = Bun.spawn([process.execPath, "run", "scripts/http-smoke.ts"], {
        cwd: fileURLToPath(new URL("..", import.meta.url)),
        env: { ...process.env, LOCALJEV_URL: server.url.href },
        stdout: "pipe", stderr: "pipe",
      });
      expect(await child.exited).toBe(1);
      expect(await new Response(child.stderr).text()).toContain("HTTP smoke failed: ready returned HTTP 503");
      expect(await new Response(child.stdout).text()).toBe("");
    } finally {
      await server.stop(true);
    }
  });
});
