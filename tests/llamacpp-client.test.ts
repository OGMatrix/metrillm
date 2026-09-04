/**
 * Test intent:
 * - Cover the llama.cpp (llama-server) runtime client: model listing,
 *   non-streaming and streaming chat parsing, thinking handling, option
 *   negotiation, timeouts and abort behavior.
 *
 * Why it matters:
 * - llama.cpp is a first-class backend; its OpenAI-compatible responses
 *   must map onto the shared GenerateResult contract (tok/s, TTFT, tokens).
 */
import { beforeEach, describe, expect, it, vi } from "vitest";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function chatCompletionResponse(
  response: string,
  options?: {
    reasoning?: string;
    promptTokens?: number;
    completionTokens?: number;
    predictedPerSecond?: number;
    promptMs?: number;
    predictedMs?: number;
  }
): Response {
  const timings: Record<string, number> = {};
  if (options?.predictedPerSecond !== undefined) timings.predicted_per_second = options.predictedPerSecond;
  if (options?.promptMs !== undefined) timings.prompt_ms = options.promptMs;
  if (options?.predictedMs !== undefined) timings.predicted_ms = options.predictedMs;
  return jsonResponse({
    id: "chatcmpl-1",
    object: "chat.completion",
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          content: response,
          ...(options?.reasoning ? { reasoning_content: options.reasoning } : {}),
        },
        finish_reason: "stop",
      },
    ],
    usage: {
      prompt_tokens: options?.promptTokens ?? 3,
      completion_tokens: options?.completionTokens ?? 1,
      total_tokens: (options?.promptTokens ?? 3) + (options?.completionTokens ?? 1),
    },
    ...(Object.keys(timings).length > 0 ? { timings } : {}),
  });
}

function chatStreamResponse(chunks: Array<Record<string, unknown>>): Response {
  const encoder = new TextEncoder();
  const sse = [
    ...chunks.map((chunk) => `data: ${JSON.stringify(chunk)}`),
    "data: [DONE]",
  ].join("\n\n") + "\n\n";
  return new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(encoder.encode(sse));
        controller.close();
      },
    }),
    {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    }
  );
}

function slowChatStreamResponse(chunkDelayMs: number): Response {
  const encoder = new TextEncoder();
  return new Response(
    new ReadableStream({
      async start(controller) {
        await new Promise((resolve) => setTimeout(resolve, chunkDelayMs));
        controller.enqueue(encoder.encode('data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'));
        controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        controller.close();
      },
    }),
    {
      status: 200,
      headers: { "Content-Type": "text/event-stream" },
    }
  );
}

function requestPath(input: RequestInfo | URL): string {
  if (input instanceof URL) return input.pathname;
  if (typeof input === "string") return new URL(input).pathname;
  if (input instanceof Request) return new URL(input.url).pathname;
  return "";
}

describe("llamacpp-client model discovery", () => {
  beforeEach(() => {
    vi.resetModules();
    delete process.env.LLAMA_CPP_BASE_URL;
    delete process.env.LLAMA_CPP_API_KEY;
    delete process.env.METRILLM_STREAM_STALL_TIMEOUT_MS;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("maps /v1/models entries to OllamaModel entries", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
      expect(requestPath(input)).toBe("/v1/models");
      return jsonResponse({
        object: "list",
        data: [
          {
            id: "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            object: "model",
            meta: { n_params: 8030261312, size: 4912898304 },
          },
          {
            id: "../models/phi-3-mini.gguf",
            object: "model",
            meta: null,
          },
        ],
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    const models = await client.listModels();

    expect(models).toEqual([
      {
        name: "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        size: 4912898304,
        parameterSize: "8B",
        quantization: undefined,
        modelFormat: "gguf",
      },
      {
        name: "../models/phi-3-mini.gguf",
        size: 0,
        parameterSize: undefined,
        quantization: undefined,
        modelFormat: "gguf",
      },
    ]);
  });

  it("surfaces a readable error when the model list request fails", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ error: { message: "server not ready" } }, 503)
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    await expect(client.listModels()).rejects.toThrow(
      /llama\.cpp list models failed \(503.*server not ready\)/i
    );
  });

  it("reports running models from the loaded model list", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        object: "list",
        data: [{ id: "model.gguf", meta: { size: 1000, n_params: 350000000 } }],
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    const running = await client.listRunningModels();

    expect(running).toEqual([{ name: "model.gguf", size: 1000, vramUsed: 0 }]);
  });

  it("reads the server build info from /props and falls back to unknown", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ build_info: "b3123-abcdef0123", model_path: "model.gguf" })
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    expect(await client.getLlamaCppVersion()).toBe("b3123-abcdef0123");

    const failingClientFetch = vi.fn(async () => jsonResponse({ error: "nope" }, 404));
    vi.stubGlobal("fetch", failingClientFetch);
    expect(await client.getLlamaCppVersion()).toBe("unknown");
  });

  it("uses LLAMA_CPP_BASE_URL and LLAMA_CPP_API_KEY when set", async () => {
    process.env.LLAMA_CPP_BASE_URL = "http://192.168.10.50:8080";
    process.env.LLAMA_CPP_API_KEY = "sk-test";
    const fetchMock = vi.fn(async () =>
      jsonResponse({ object: "list", data: [] })
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    await client.listModels();

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const [input, init] = fetchMock.mock.calls[0];
    expect(new URL(String(input)).toString()).toBe("http://192.168.10.50:8080/v1/models");
    expect((init?.headers as Record<string, string>).Authorization).toBe("Bearer sk-test");
  });
});

describe("llamacpp-client non-streaming chat", () => {
  beforeEach(() => {
    vi.resetModules();
    delete process.env.LLAMA_CPP_BASE_URL;
    delete process.env.LLAMA_CPP_API_KEY;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("sends benchmark sampling and max_tokens, and maps usage/timings", async () => {
    let capturedBody: Record<string, unknown> | null = null;
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const rawBody = typeof init?.body === "string" ? init.body : "{}";
      capturedBody = JSON.parse(rawBody) as Record<string, unknown>;
      return chatCompletionResponse("Hello world", {
        promptTokens: 12,
        completionTokens: 24,
        predictedPerSecond: 48,
        promptMs: 150,
        predictedMs: 500,
      });
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    const result = await client.generate("model-a", "prompt", { temperature: 0, top_p: 1, seed: 42, num_predict: 256 });

    expect(capturedBody).toMatchObject({
      model: "model-a",
      messages: [{ role: "user", content: "prompt" }],
      stream: false,
      temperature: 0,
      top_p: 1,
      seed: 42,
      max_tokens: 256,
    });
    expect(capturedBody).not.toHaveProperty("chat_template_kwargs");
    expect(capturedBody).not.toHaveProperty("stream_options");

    expect(result.response).toBe("Hello world");
    expect(result.promptEvalCount).toBe(12);
    expect(result.evalCount).toBe(24);
    expect(result.promptEvalDuration).toBe(150_000_000);
    expect(result.evalDuration).toBe(500_000_000);
    expect(result.evalCountEstimated).toBeUndefined();
  });

  it("sends enable_thinking=false and captures reasoning_content when think=false", async () => {
    let capturedBody: Record<string, unknown> | null = null;
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const rawBody = typeof init?.body === "string" ? init.body : "{}";
      capturedBody = JSON.parse(rawBody) as Record<string, unknown>;
      return chatCompletionResponse("Final answer", { reasoning: "hidden thoughts" });
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    const result = await client.generate("model-a", "prompt", { think: false });

    expect(capturedBody).toMatchObject({
      chat_template_kwargs: { enable_thinking: false },
    });
    expect(result.response).toBe("Final answer");
    expect(result.thinking).toBe("hidden thoughts");
  });

  it("fails fast when non-thinking mode still returns thinking content", async () => {
    const fetchMock = vi.fn(async () =>
      chatCompletionResponse("[THINK]I should reason first[/THINK]Final answer")
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    await expect(
      client.generate("model-a", "prompt", { think: false })
    ).rejects.toThrow(/still emitted thinking content/i);
  });

  it("retries without optional extras when the backend rejects them", async () => {
    const capturedBodies: Record<string, unknown>[] = [];
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const rawBody = typeof init?.body === "string" ? init.body : "{}";
      const parsed = JSON.parse(rawBody) as Record<string, unknown>;
      capturedBodies.push(parsed);
      if (capturedBodies.length === 1) {
        return jsonResponse({ error: { message: "unknown field: chat_template_kwargs" } }, 400);
      }
      return chatCompletionResponse("OK");
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    const result = await client.generate("model-a", "prompt", { think: false });

    expect(result.response).toBe("OK");
    expect(capturedBodies).toHaveLength(2);
    expect(capturedBodies[0]).toHaveProperty("chat_template_kwargs");
    expect(capturedBodies[1]).not.toHaveProperty("chat_template_kwargs");
  });

  it("retries without top_p/seed when the backend rejects sampling options", async () => {
    const capturedBodies: Record<string, unknown>[] = [];
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const rawBody = typeof init?.body === "string" ? init.body : "{}";
      const parsed = JSON.parse(rawBody) as Record<string, unknown>;
      capturedBodies.push(parsed);
      if (capturedBodies.length === 1) {
        return jsonResponse({ error: { message: "Unrecognized key(s) in object: 'seed'" } }, 400);
      }
      return chatCompletionResponse("OK");
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    const result = await client.generate("model-a", "prompt", { top_p: 1, seed: 42 });

    expect(result.response).toBe("OK");
    expect(capturedBodies).toHaveLength(2);
    expect(capturedBodies[0]).toMatchObject({ top_p: 1, seed: 42 });
    expect(capturedBodies[1]).not.toHaveProperty("top_p");
    expect(capturedBodies[1]).not.toHaveProperty("seed");
  });

  it("falls back to token estimation when usage is missing", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({
        choices: [{ message: { role: "assistant", content: "one two three four five" } }],
      })
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    const result = await client.generate("model-a", "prompt");

    expect(result.response).toBe("one two three four five");
    expect(result.evalCount).toBeGreaterThanOrEqual(5);
    expect(result.evalCountEstimated).toBe(true);
  });
});

describe("llamacpp-client streaming chat", () => {
  beforeEach(() => {
    vi.resetModules();
    delete process.env.LLAMA_CPP_BASE_URL;
    delete process.env.LLAMA_CPP_API_KEY;
    delete process.env.METRILLM_STREAM_STALL_TIMEOUT_MS;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("accumulates streamed deltas and reads usage/timings from the final chunk", async () => {
    const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      const rawBody = typeof init?.body === "string" ? init.body : "{}";
      expect(JSON.parse(rawBody)).toMatchObject({ stream: true });
      return chatStreamResponse([
        { choices: [{ delta: { role: "assistant" } }] },
        { choices: [{ delta: { content: "Hello " } }] },
        { choices: [{ delta: { content: "world" } }] },
        {
          choices: [{ delta: {} }],
          usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
          timings: { predicted_ms: 250, predicted_per_second: 8, prompt_ms: 100 },
        },
      ]);
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    const tokens: string[] = [];
    const result = await client.generateStream(
      "model-a",
      "prompt",
      { onToken: (token) => tokens.push(token) }
    );

    expect(result.response).toBe("Hello world");
    expect(tokens).toEqual(["Hello ", "world"]);
    expect(result.promptEvalCount).toBe(10);
    expect(result.evalCount).toBe(2);
    expect(result.promptEvalDuration).toBe(100_000_000);
    expect(result.evalDuration).toBe(250_000_000);
    expect(result.evalCountEstimated).toBeUndefined();
  });

  it("captures streamed reasoning_content as thinking", async () => {
    const fetchMock = vi.fn(async () =>
      chatStreamResponse([
        { choices: [{ delta: { reasoning_content: "planning " } }] },
        { choices: [{ delta: { reasoning_content: "done" } }] },
        { choices: [{ delta: { content: "Answer" } }] },
      ])
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    const result = await client.generateStream("model-a", "prompt");

    expect(result.response).toBe("Answer");
    expect(result.thinking).toBe("planning done");
  });

  it("falls back to estimated token counts when the stream has no usage", async () => {
    const fetchMock = vi.fn(async () =>
      chatStreamResponse([
        { choices: [{ delta: { content: "a b c d" } }] },
      ])
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    const result = await client.generateStream("model-a", "prompt");

    expect(result.evalCount).toBeGreaterThanOrEqual(4);
    expect(result.evalCountEstimated).toBe(true);
  });

  it("aborts the stream when no chunk arrives within the shared stall timeout", async () => {
    process.env.METRILLM_STREAM_STALL_TIMEOUT_MS = "50";
    const fetchMock = vi.fn(async () => slowChatStreamResponse(2_000));
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    await expect(client.generateStream("model-a", "prompt")).rejects.toThrow(
      /llama\.cpp stream timed out after 50ms/i
    );
  });

  it("reports an error when the stream ends without any content", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(
        new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode("data: [DONE]\n\n"));
            controller.close();
          },
        }),
        { status: 200, headers: { "Content-Type": "text/event-stream" } }
      )
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    await expect(client.generateStream("model-a", "prompt")).rejects.toThrow(
      /llama\.cpp stream ended without content/i
    );
  });
});

describe("llamacpp-client lifecycle", () => {
  beforeEach(() => {
    vi.resetModules();
    delete process.env.LLAMA_CPP_BASE_URL;
    delete process.env.LLAMA_CPP_API_KEY;
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("unloads a model via /models/unload for router-mode servers", async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      expect(requestPath(input)).toBe("/models/unload");
      expect(init?.method).toBe("POST");
      expect(init?.body).toBe(JSON.stringify({ model: "model.gguf" }));
      return jsonResponse({ success: true });
    });
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    await client.unloadModel("model.gguf");
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("treats unload failures as a no-op on regular servers", async () => {
    const fetchMock = vi.fn(async () =>
      jsonResponse({ error: { message: "not found" } }, 404)
    );
    vi.stubGlobal("fetch", fetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    await expect(client.unloadModel("model.gguf")).resolves.toBeUndefined();
  });

  it("aborts in-flight requests on abortOngoingRequests", async () => {
    const fetchMock = vi.fn(
      () =>
        new Promise<Response>((_resolve, reject) => {
          // Keep the request pending until the abort signal fires.
          const timer = setTimeout(() => reject(new Error("unexpected timeout")), 5_000);
          // The client passes an AbortSignal; wire it manually here is not possible,
          // so resolve the promise when aborted via the controller the client created.
          void timer;
          return;
        })
    );
    // Instead of faking a pending fetch, verify the abort wiring through a slow
    // stream that can actually observe the signal: the client aborts the fetch
    // when abortOngoingRequests() is called.
    vi.unstubAllGlobals();
    delete fetchMock.mock.calls;

    let fetchSignal: AbortSignal | null = null;
    const realFetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      fetchSignal = init?.signal ?? null;
      return slowChatStreamResponse(2_000);
    });
    vi.stubGlobal("fetch", realFetchMock);

    const client = await import("../src/core/llamacpp-client.js");
    const streamPromise = client.generateStream("model-a", "prompt");
    // Wait for the stream to start, then abort.
    await new Promise((resolve) => setTimeout(resolve, 20));
    client.abortOngoingRequests();

    await expect(streamPromise).rejects.toThrow(/aborted|timed out/i);
    expect(fetchSignal?.aborted).toBe(true);
  });
});
