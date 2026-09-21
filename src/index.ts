import { loadSettings } from "./config";
import { Engine } from "./engine";
import { LocalJevApp } from "./server";

export { loadSettings } from "./config";
export { Engine } from "./engine";
export { LocalJevApp } from "./server";
export * from "./types";

export function serveLocalJev(app: LocalJevApp, idleTimeout = 255) {
  return Bun.serve({
    hostname: app.settings.host,
    port: app.settings.port,
    idleTimeout,
    fetch(request, server) {
      if (
        request.method === "POST" &&
        new URL(request.url).pathname === "/v1/systemone"
      ) {
        // Inference and validation retries can exceed Bun's idle timeout.
        // The engine enforces the configured timeout on each upstream call.
        server.timeout(request, 0);
      }
      return app.fetch(request);
    },
  });
}

if (import.meta.main) {
  const settings = loadSettings();
  const app = new LocalJevApp(settings, new Engine(settings));
  const server = serveLocalJev(app);

  console.log(`LocalJev listening on ${server.url}`);

  let closing = false;
  const shutdown = async () => {
    if (closing) return;
    closing = true;
    await server.stop();
    await app.close();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}
