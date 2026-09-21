import { expect, test } from "bun:test";

import { loadSettings, LocalJevApp, serveLocalJev } from "../src/index";
import type { DecisionEngine } from "../src/engine";

test("inference can outlast the HTTP idle timeout", async () => {
  const engine: DecisionEngine = {
    async decide() {
      // Bun checks a one-second idle limit on a roughly four-second timer.
      await Bun.sleep(5_500);
      return {
        answers: { urgent: { type: "noul", noul: 0.9 } },
        inputTokens: 10,
        outputTokens: 5,
      };
    },
  };
  const app = new LocalJevApp(
    loadSettings({ host: "127.0.0.1", port: 0, apiKey: "" }),
    engine,
  );
  const server = serveLocalJev(app, 1);
  try {
    const response = await fetch(new URL("v1/systemone", server.url), {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        model: "localjev-latest",
        state: "A production service is down.",
        questions: { urgent: { type: "noul", instructions: "Is this urgent?" } },
      }),
      signal: AbortSignal.timeout(10_000),
    });
    expect(response.status).toBe(200);
    expect((await response.json()).answers.urgent).toEqual({ type: "noul", noul: 0.9 });
  } finally {
    await server.stop(true);
    await app.close();
  }
}, 12_000);
