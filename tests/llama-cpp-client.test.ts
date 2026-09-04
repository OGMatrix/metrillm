import { beforeEach, describe, expect, it, vi } from "vitest";

function makeSseResponse(chunks: Array<Record<string, unknown>>): Response {
  const encoder = new TextEncoder();
  const body = chunks
    .map((chunk) => `data: ${JSON.stringify(chunk)}\n\n`)
    .concat(["data: [DONE]\n\n"])
    .join("");
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode(body));
        controller.close();
      },
    }),
    { status: 200, headers: { "Content-Type": "text/event-stream" } }
  );
}

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("llama-cpp-client", () => {
  beforeEach(() => {
    vi.resetModules();
    vi.unstubAllGlobals();
    delete process.env.LLAMA_CPP_BASE_URL;
    delete process.env.LLAMA_CPP_API_KEY;
    delete process.env.METRILLM_STREAM_STALL_TIMEOUT_MS;
  });

  describe("getLlamaCppVersion", () => {
    it("reads build_info from /props", async () => {
      vi.stubGlobal("fetch", vi.fn(async (url: string) => {
        expect(url).toContain("/props");
        return jsonResponse({ build_info: "b4993 (abcdef)" });
      }));

      const client = await import("../src/core/llama-cpp-client.js");
      await expect(client.getLlamaCppVersion()).resolves.toBe("b4993 (abcdef)");
    });

    it("returns unknown when the server is unreachable", async () => {
      vi.stubGlobal("fetch", vi.fn(async () => {
        throw new Error("ECONNREFUSED");
      }));

      const client = await import("../src/core/llama-cpp-client.js");
      await expect(client.getLlamaCppVersion()).resolves.toBe("unknown");
    });
  });

  describe("listModels", () => {
    it("prefers router /models when available (with status)", async () => {
      const fetchMock = vi.fn(async (url: string) => {
        if (url.endsWith("/v1/models")) return jsonResponse({ data: [] });
        if (url.endsWith("/models")) {
          return jsonResponse({
            data: [
              {
                id: "ggml-org/Qwen3-8B-GGUF:Q4_K_M",
                path: "/models/qwen3.gguf",
                status: { value: "loaded" },
                meta: { n_params: 8094912000, size: 5127000000 },
              },
              {
                id: "ggml-org/Llama-3.2-1B-GGUF:Q8_0",
                status: { value: "unloaded" },
                meta: { n_params: 1248173568 },
              },
            ],
          });
        }
        throw new Error(`unexpected url ${url}`);
      });
      vi.stubGlobal("fetch", fetchMock);

      const client = await import("../src/core/llama-cpp-client.js");
      const models = await client.listModels();

      expect(models).toHaveLength(2);
      expect(models[0]).toMatchObject({
        name: "ggml-org/Qwen3-8B-GGUF:Q4_K_M",
        size: 5127000000,
        parameterSize: "8.1B",
        quantization: "Q4_K_M",
        modelFormat: "gguf",
        runtimeStatus: "loaded",
      });
      expect(models[1]).toMatchObject({
        name: "ggml-org/Llama-3.2-1B-GGUF:Q8_0",
        size: 0,
        parameterSize: "1.2B",
        quantization: "Q8_0",
        runtimeStatus: "unloaded",
      });
    });

    it("falls back to /v1/models for single-model servers and infers quant from id", async () => {
      const fetchMock = vi.fn(async (url: string) => {
        if (url.endsWith("/v1/models")) {
          return jsonResponse({
            object: "list",
            data: [
              { id: "Qwen3-8B-Q4_K_M.gguf", meta: { n_params: 8094912000 } },
              { id: "mistral-7b-instruct-v0.3" },
            ],
          });
        }
        // Non-router: /models should not exist → 404
        return jsonResponse({ error: "not found" }, 404);
      });
      vi.stubGlobal("fetch", fetchMock);

      const client = await import("../src/core/llama-cpp-client.js");
      const models = await client.listModels();

      expect(models).toHaveLength(2);
      expect(models[0]).toMatchObject({
        name: "Qwen3-8B-Q4_K_M.gguf",
        parameterSize: "8.1B",
        quantization: "Q4_K_M",
      });
      expect(models[1]).toMatchObject({
        name: "mistral-7b-instruct-v0.3",
        parameterSize: "7B",
        quantization: undefined,
      });
    });

    it("throws with status on /v1/models failure when router is unavailable", async () => {
      vi.stubGlobal("fetch", vi.fn(async (url: string) => {
        if (url.endsWith("/v1/models")) {
          return new Response("bad gateway", { status: 502 });
        }
        return new Response("nope", { status: 404 });
      }));

      const client = await import("../src/core/llama-cpp-client.js");
      await expect(client.listModels()).rejects.toThrow(/llama\.cpp list models failed \(502/);
    });
  });

  describe("listRunningModels", () => {
    it("filters router models to loaded status", async () => {
      const fetchMock = vi.fn(async (url: string) => {
        if (url.endsWith("/v1/models")) {
          return jsonResponse({
            data: [{ id: "A.gguf", meta: { size: 1 } }, { id: "B.gguf", meta: { size: 2 } }],
          });
        }
        return jsonResponse({
          data: [
            { id: "A.gguf", status: { value: "loaded" }, meta: { size: 1234 } },
            { id: "B.gguf", status: { value: "unloaded" }, meta: { size: 2222 } },
          ],
        });
      });
      vi.stubGlobal("fetch", fetchMock);

      const client = await import("../src/core/llama-cpp-client.js");
      const running = await client.listRunningModels();
      expect(running).toEqual([{ name: "A.gguf", size: 1234, vramUsed: 0 }]);
    });

    it("treats the single served model as running without router", async () => {
      const fetchMock = vi.fn(async (url: string) => {
        if (url.endsWith("/v1/models")) {
          return jsonResponse({ data: [{ id: "only.gguf" }] });
        }
        return new Response("nope", { status: 404 });
      });
      vi.stubGlobal("fetch", fetchMock);

      const client = await import("../src/core/llama-cpp-client.js");
      const running = await client.listRunningModels();
      expect(running).toEqual([{ name: "only.gguf", size: 0, vramUsed: 0 }]);
    });
  });

  describe("generateStream", () => {
    it("maps usage and timings from the final chunk into GenerateResult", async () => {
      const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
        if (url.endsWith("/v1/chat/completions")) {
          const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
          expect(body.model).toBe("test-model");
          expect(body.stream).toBe(true);
          expect(body.max_tokens).toBe(64);
          expect(body.seed).toBe(42);
          expect(body.temperature).toBe(0);
          expect(body.stream_options).toEqual({ include_usage: true });
          return makeSseResponse([
            { choices: [{ index: 0, delta: { role: "assistant", content: "" } }] },
            { choices: [{ index: 0, delta: { content: "Hello " } }] },
            { choices: [{ index: 0, delta: { content: "world" } }] },
            {
              choices: [{ index: 0, delta: {}, finish_reason: "length" }],
              usage: { prompt_tokens: 12, completion_tokens: 64, total_tokens: 76 },
              timings: {
                prompt_n: 12,
                cache_n: 0,
                prompt_ms: 300,
                predicted_n: 64,
                predicted_ms: 4096,
              },
            },
          ]);
        }
        throw new Error(`unexpected url ${url}`);
      });
      vi.stubGlobal("fetch", fetchMock);

      const client = await import("../src/core/llama-cpp-client.js");
      const tokens: string[] = [];
      const result = await client.generateStream(
        "test-model",
        "Say hello",
        { onToken: (t) => tokens.push(t) },
        { temperature: 0, top_p: 1, seed: 42, num_predict: 64 }
      );

      expect(result.response).toBe("Hello world");
      expect(tokens).toEqual(["Hello ", "world"]);
      expect(result.promptEvalCount).toBe(12);
      expect(result.evalCount).toBe(64);
      expect(result.promptEvalDuration).toBe(300_000_000); // 300ms → ns
      expect(result.evalDuration).toBe(4_096_000_000); // 4096ms → ns
      expect(result.evalCountEstimated).toBeUndefined();
    });

    it("retries without sampling options when the server rejects them", async () => {
      let attempts = 0;
      const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
        if (url.endsWith("/v1/chat/completions")) {
          attempts++;
          const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
          if (attempts === 1 && body.seed !== undefined) {
            return new Response(
              JSON.stringify({ error: { message: "unknown field: seed", type: "invalid_request_error", code: 400 } }),
              { status: 400 }
            );
          }
          return makeSseResponse([
            { choices: [{ index: 0, delta: { content: "ok" } }] },
            {
              choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
              usage: { prompt_tokens: 3, completion_tokens: 1, total_tokens: 4 },
            },
          ]);
        }
        throw new Error(`unexpected url ${url}`);
      });
      vi.stubGlobal("fetch", fetchMock);

      const client = await import("../src/core/llama-cpp-client.js");
      const result = await client.generateStream("m", "p", undefined, { seed: 42, top_p: 1 });

      expect(result.response).toBe("ok");
      expect(attempts).toBe(2);
      const secondBody = JSON.parse(
        String(fetchMock.mock.calls[1]?.[1]?.body)
      ) as Record<string, unknown>;
      expect(secondBody.seed).toBeUndefined();
      expect(secondBody.top_p).toBeUndefined();
    });

    it("estimates eval count when no usage/timings are reported", async () => {
      vi.stubGlobal("fetch", vi.fn(async (url: string) => {
        if (url.endsWith("/v1/chat/completions")) {
          return makeSseResponse([
            { choices: [{ index: 0, delta: { content: "a b c d e f g h i j" } }] },
            { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
          ]);
        }
        throw new Error(`unexpected url ${url}`);
      }));

      const client = await import("../src/core/llama-cpp-client.js");
      const result = await client.generateStream("m", "p");

      expect(result.evalCount).toBeGreaterThan(0);
      expect(result.evalCountEstimated).toBe(true);
    });

    it("captures reasoning_content separately from response", async () => {
      vi.stubGlobal("fetch", vi.fn(async (url: string) => {
        if (url.endsWith("/v1/chat/completions")) {
          return makeSseResponse([
            { choices: [{ index: 0, delta: { reasoning_content: "Let me think..." } }] },
            { choices: [{ index: 0, delta: { content: "42" } }] },
            {
              choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
              usage: { prompt_tokens: 5, completion_tokens: 7, total_tokens: 12 },
            },
          ]);
        }
        throw new Error(`unexpected url ${url}`);
      }));

      const client = await import("../src/core/llama-cpp-client.js");
      const result = await client.generateStream("m", "What is 6*7?");

      expect(result.response).toBe("42");
      expect(result.thinking).toBe("Let me think...");
    });

    it("honours the shared stream stall timeout environment variable", async () => {
      const setTimeoutSpy = vi.spyOn(globalThis, "setTimeout");
      process.env.METRILLM_STREAM_STALL_TIMEOUT_MS = "2345";
      try {
        vi.stubGlobal("fetch", vi.fn(async (url: string) => {
          if (url.endsWith("/v1/chat/completions")) {
            return makeSseResponse([
              { choices: [{ index: 0, delta: { content: "x" } }] },
              { choices: [{ index: 0, delta: {}, finish_reason: "stop" }] },
            ]);
          }
          throw new Error(`unexpected url ${url}`);
        }));

        const client = await import("../src/core/llama-cpp-client.js");
        await client.generateStream("m", "p");

        expect(setTimeoutSpy.mock.calls.some((call) => call[1] === 2345)).toBe(true);
      } finally {
        setTimeoutSpy.mockRestore();
        delete process.env.METRILLM_STREAM_STALL_TIMEOUT_MS;
      }
    });
  });

  describe("generate", () => {
    it("reads message + usage + timings from a non-streamed response", async () => {
      vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
        if (url.endsWith("/v1/chat/completions")) {
          const body = JSON.parse(String(init?.body)) as Record<string, unknown>;
          expect(body.stream).toBe(false);
          expect(body.stream_options).toBeUndefined();
          return jsonResponse({
            object: "chat.completion",
            choices: [{ index: 0, message: { role: "assistant", content: "Hi" }, finish_reason: "stop" }],
            usage: { prompt_tokens: 4, completion_tokens: 1, total_tokens: 5 },
            timings: { prompt_n: 4, cache_n: 1, prompt_ms: 120, predicted_n: 1, predicted_ms: 50 },
          });
        }
        throw new Error(`unexpected url ${url}`);
      }));

      const client = await import("../src/core/llama-cpp-client.js");
      const result = await client.generate("m", "p");

      expect(result.response).toBe("Hi");
      expect(result.promptEvalCount).toBe(4);
      expect(result.evalCount).toBe(1);
      expect(result.evalDuration).toBe(50_000_000); // 50ms → ns
    });
  });

  describe("unloadModel", () => {
    it("calls /models/unload and tolerates 404/405 from single-model servers", async () => {
      const fetchMock = vi.fn(async (url: string, init?: RequestInit) => {
        expect(url).toContain("/models/unload");
        expect(JSON.parse(String(init?.body))).toEqual({ model: "test" });
        return jsonResponse({ success: true });
      });
      vi.stubGlobal("fetch", fetchMock);

      const client = await import("../src/core/llama-cpp-client.js");
      await expect(client.unloadModel("test")).resolves.toBeUndefined();

      vi.stubGlobal("fetch", vi.fn(async () => new Response("not found", { status: 404 })));
      await expect(client.unloadModel("test")).resolves.toBeUndefined();
    });

    it("throws on other failures", async () => {
      vi.stubGlobal("fetch", vi.fn(async () =>
        new Response(JSON.stringify({ error: "boom" }), { status: 500 })
      ));

      const client = await import("../src/core/llama-cpp-client.js");
      await expect(client.unloadModel("test")).rejects.toThrow(/llama\.cpp unload failed \(500/);
    });
  });

  describe("connection config", () => {
    it("uses LLAMA_CPP_BASE_URL and LLAMA_CPP_API_KEY when set", async () => {
      process.env.LLAMA_CPP_BASE_URL = "192.168.1.50:9090";
      process.env.LLAMA_CPP_API_KEY = "secret-key";
      const fetchMock = vi.fn(async () =>
        jsonResponse({
          data: [
            { id: "one", status: { value: "loaded" }, meta: { size: 100 } },
          ],
        })
      );
      vi.stubGlobal("fetch", fetchMock);

      const client = await import("../src/core/llama-cpp-client.js");
      const models = await client.listModels();
      expect(models).toHaveLength(1);

      const firstCall = fetchMock.mock.calls[0];
      const url = String(firstCall?.[0]);
      expect(url.startsWith("http://192.168.1.50:9090/")).toBe(true);
      const headers = firstCall?.[1]?.headers as Record<string, string>;
      expect(headers.Authorization).toBe("Bearer secret-key");
    });
  });
});
