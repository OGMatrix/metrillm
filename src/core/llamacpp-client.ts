import type { OllamaModel, OllamaRunningModel } from "../types.js";
import type { GenerateResult, KeepAliveValue, StreamCallbacks } from "./ollama-client.js";
import { estimateTokenCount } from "../utils.js";

const DEFAULT_LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080";
const LLAMA_CPP_INIT_TIMEOUT_MS = 15_000;
const LLAMA_CPP_METADATA_TIMEOUT_MS = 2_000;
const DEFAULT_STREAM_STALL_TIMEOUT_MS = 30_000;
const SHARED_STREAM_STALL_TIMEOUT_ENV = "METRILLM_STREAM_STALL_TIMEOUT_MS";
const LLAMA_CPP_BASE_URL_ENV = "LLAMA_CPP_BASE_URL";
const LLAMA_CPP_API_KEY_ENV = "LLAMA_CPP_API_KEY";

interface LlamaCppRequestOptions {
  temperature?: number;
  top_p?: number;
  seed?: number;
  num_predict?: number;
  keep_alive?: KeepAliveValue;
  think?: boolean;
  stall_timeout_ms?: number;
}

interface LlamaCppChatCompletionChoice {
  message?: {
    content?: string;
    reasoning_content?: string;
  };
}

interface LlamaCppChatCompletionResponse {
  choices?: LlamaCppChatCompletionChoice[];
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  timings?: {
    prompt_n?: number;
    prompt_ms?: number;
    predicted_n?: number;
    predicted_ms?: number;
    predicted_per_second?: number;
  };
}

interface LlamaCppChatChunkDelta {
  role?: string;
  content?: string;
  reasoning_content?: string;
}

interface LlamaCppChatCompletionChunk {
  choices?: Array<{
    delta?: LlamaCppChatChunkDelta;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
  timings?: {
    prompt_n?: number;
    prompt_ms?: number;
    predicted_n?: number;
    predicted_ms?: number;
    predicted_per_second?: number;
  };
}

interface LlamaCppModelListResponse {
  data?: Array<{
    id?: string;
    meta?: {
      n_params?: number;
      size?: number;
    } | null;
  }>;
}

interface LlamaCppModelListEntry {
  id?: string;
  status?: {
    value?: string;
  };
}

interface LlamaCppModelsListResponse {
  data?: LlamaCppModelListEntry[];
}

interface LlamaCppPropsResponse {
  build_info?: string;
  model_path?: string;
}

let defaultKeepAlive: KeepAliveValue | undefined;
const activeAbortControllers = new Set<AbortController>();
const unsupportedOptionsCache = new Map<string, boolean>();

const NON_THINKING_CHAT_TEMPLATE_KWARGS = { enable_thinking: false };

function hasThinkingLeakText(response: string): boolean {
  return (
    /^\s*(?:thinking|thought)\s+process\s*:/i.test(response)
    || /\[(?:\/)?THINK(?:ING)?\]/i.test(response)
  );
}

function assertThinkingModeRespected(
  model: string,
  think: boolean | undefined,
  response: string,
  reasoning: string
): void {
  if (think !== false) return;
  if (reasoning.trim().length > 0 || /<think(?:ing)?[\s>]/i.test(response) || hasThinkingLeakText(response)) {
    throw new Error(
      [
        `llama.cpp model "${model}" still emitted thinking content while non-thinking mode is requested.`,
        "chat_template_kwargs {\"enable_thinking\": false} is already sent by MetriLLM, so this model's template likely has no non-thinking switch.",
        "Benchmark this model with --thinking, or serve a model with a thinking-capable template.",
      ].join(" ")
    );
  }
}

const UNSUPPORTED_OPTION_FIELD_PATTERN = /\b(seed|top_p|topp|stream_options|chat_template_kwargs)\b/;

function isUnsupportedOptionMessage(status: number, text: string): boolean {
  if (status !== 400 && status !== 422) return false;
  const lower = text.toLowerCase();
  if (!UNSUPPORTED_OPTION_FIELD_PATTERN.test(lower)) return false;
  return /unrecognized|unknown|not support|unsupported|invalid|unexpected|additional|extra/.test(lower);
}

function extractLlamaCppErrorMessage(body: string): string {
  const trimmed = body.trim();
  if (!trimmed) return "";
  try {
    const parsed = JSON.parse(trimmed) as {
      error?: {
        message?: unknown;
      } | string;
    };
    const error = parsed.error;
    if (typeof error === "string" && error.trim().length > 0) {
      return error.trim();
    }
    const message = typeof error === "object" && error !== null ? error.message : undefined;
    if (typeof message === "string" && message.trim().length > 0) {
      return message.trim();
    }
  } catch {
    // Fall back to raw response body text when payload is not JSON.
  }
  return trimmed;
}

function buildLlamaCppRequestError(
  kind: "generate" | "stream",
  model: string,
  status: number,
  statusText: string,
  body: string
): Error {
  const backendMessage = extractLlamaCppErrorMessage(body);
  const suffix = backendMessage ? ` ${backendMessage}` : "";
  return new Error(`llama.cpp ${kind} failed (${status} ${statusText})${suffix}`.trim());
}

function hasSamplingOverrides(options?: LlamaCppRequestOptions): boolean {
  return options?.top_p !== undefined || options?.seed !== undefined;
}

function buildChatBody(
  model: string,
  prompt: string,
  options: LlamaCppRequestOptions | undefined,
  includeExtras: boolean,
  stream: boolean
): Record<string, unknown> {
  const body: Record<string, unknown> = {
    model,
    messages: [{ role: "user", content: prompt }],
    stream,
    temperature: options?.temperature ?? 0,
    max_tokens: options?.num_predict ?? 512,
  };
  if (includeExtras) {
    if (options?.top_p !== undefined) body.top_p = options.top_p;
    if (options?.seed !== undefined) body.seed = options.seed;
    if (options?.think === false) body.chat_template_kwargs = { ...NON_THINKING_CHAT_TEMPLATE_KWARGS };
    if (stream) body.stream_options = { include_usage: true };
  }
  return body;
}

function negotiateChatRequest(
  kind: "generate" | "stream",
  model: string,
  cacheKey: string,
  options: LlamaCppRequestOptions | undefined,
  makeRequest: (includeExtras: boolean) => Promise<Response>
): Promise<Response> {
  const runRequest = async (includeExtras: boolean) => {
    const resp = await makeRequest(includeExtras);
    if (!resp.ok) {
      const body = await resp.text().catch(() => "");
      if (includeExtras && isUnsupportedOptionMessage(resp.status, body)) {
        // Some llama.cpp builds reject extra body fields. Retry once
        // without the optional extras while keeping deterministic sampling.
        unsupportedOptionsCache.set(cacheKey, true);
        const retryResp = await makeRequest(false);
        if (!retryResp.ok) {
          const retryBody = await retryResp.text().catch(() => "");
          throw buildLlamaCppRequestError(kind, model, retryResp.status, retryResp.statusText, retryBody);
        }
        return retryResp;
      }
      throw buildLlamaCppRequestError(kind, model, resp.status, resp.statusText, body);
    }
    return resp;
  };

  const includeExtras = hasSamplingOverrides(options) ? !unsupportedOptionsCache.get(cacheKey) : true;
  return runRequest(includeExtras);
}

function getUsageTokenCount(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) return 0;
  if (value <= 0) return 0;
  return Math.trunc(value);
}

function estimateCompletionTokensFallback(text: string): number {
  const normalized = text.trim();
  if (!normalized) return 0;

  // CJK-like scripts often have sparse/no whitespace, so whitespace tokenization
  // underestimates badly. Count those codepoints directly and estimate the rest.
  const cjkMatches = normalized.match(/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu);
  const cjkCount = cjkMatches?.length ?? 0;
  const withoutCjk = normalized.replace(
    /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Hangul}]/gu,
    ""
  );
  const nonCjkChars = withoutCjk.replace(/\s+/g, "").length;
  const nonCjkHeuristic = Math.ceil(nonCjkChars / 4);
  const whitespaceEstimate = estimateTokenCount(normalized);

  return Math.max(1, Math.max(whitespaceEstimate, cjkCount + nonCjkHeuristic));
}

function resolveCompletionTokenCount(
  reportedTokenCount: number | undefined,
  response: string,
  reasoning: string
): number {
  const reported = getUsageTokenCount(reportedTokenCount);
  if (reported > 0) return reported;
  return estimateCompletionTokensFallback(`${reasoning} ${response}`);
}

function asNonEmptyString(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function parseNonNegativeInt(value: string): number | null {
  if (!/^\d+$/.test(value)) return null;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isSafeInteger(parsed) || parsed < 0) return null;
  return parsed;
}

function resolveStreamStallTimeoutMs(override?: number): number | undefined {
  if (override !== undefined) {
    if (!Number.isFinite(override) || override < 0) return DEFAULT_STREAM_STALL_TIMEOUT_MS;
    return override === 0 ? undefined : Math.trunc(override);
  }

  const configured = process.env[SHARED_STREAM_STALL_TIMEOUT_ENV]?.trim();
  if (!configured) return DEFAULT_STREAM_STALL_TIMEOUT_MS;
  const parsed = parseNonNegativeInt(configured);
  if (parsed === null) return DEFAULT_STREAM_STALL_TIMEOUT_MS;
  return parsed === 0 ? undefined : parsed;
}

function getLlamaCppBaseUrl(): string {
  const configured = process.env[LLAMA_CPP_BASE_URL_ENV]?.trim();
  if (!configured) return DEFAULT_LLAMA_CPP_BASE_URL;
  const candidate = /^https?:\/\//i.test(configured) ? configured : `http://${configured}`;
  try {
    return new URL(candidate).toString();
  } catch {
    return DEFAULT_LLAMA_CPP_BASE_URL;
  }
}

function getLlamaCppHeaders(): Record<string, string> {
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  const apiKey = process.env[LLAMA_CPP_API_KEY_ENV]?.trim();
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }
  return headers;
}

function buildModelEntry(id: string, meta: { n_params?: number; size?: number } | null | undefined): OllamaModel {
  const parameterSize =
    typeof meta?.n_params === "number" && meta.n_params > 0
      ? `${Number.parseFloat((meta.n_params / 1e9).toPrecision(3))}B`
      : undefined;
  return {
    name: id,
    size: typeof meta?.size === "number" && meta.size > 0 ? Math.trunc(meta.size) : 0,
    parameterSize,
    quantization: undefined,
    modelFormat: "gguf",
  };
}

async function fetchLlamaCpp(
  path: string,
  init: RequestInit,
  timeoutMs: number,
  label: string
): Promise<Response> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  const baseUrl = getLlamaCppBaseUrl();
  try {
    const url = new URL(path, baseUrl);
    return await fetch(url, { ...init, signal: controller.signal });
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error(`${label} timed out after ${timeoutMs}ms`);
    }
    throw err;
  } finally {
    clearTimeout(timeout);
  }
}

export async function getLlamaCppVersion(): Promise<string> {
  try {
    const resp = await fetchLlamaCpp(
      "/props",
      { method: "GET", headers: getLlamaCppHeaders() },
      LLAMA_CPP_METADATA_TIMEOUT_MS,
      "llama.cpp version check"
    );
    if (!resp.ok) return "unknown";
    const props = (await resp.json()) as LlamaCppPropsResponse;
    return asNonEmptyString(props.build_info) ?? "unknown";
  } catch {
    return "unknown";
  }
}

export async function listModels(): Promise<OllamaModel[]> {
  const resp = await fetchLlamaCpp(
    "/v1/models",
    { method: "GET", headers: getLlamaCppHeaders() },
    LLAMA_CPP_INIT_TIMEOUT_MS,
    "llama.cpp list models"
  );
  if (!resp.ok) {
    throw new Error(`llama.cpp list models failed (${resp.status} ${resp.statusText})`);
  }
  const data = (await resp.json()) as LlamaCppModelListResponse;
  return (data.data ?? [])
    .map((m) => m.id?.trim())
    .filter((id): id is string => Boolean(id))
    .map((id) => {
      const entry = (data.data ?? []).find((m) => m.id?.trim() === id);
      return buildModelEntry(id, entry?.meta);
    });
}

export async function listRunningModels(): Promise<OllamaRunningModel[]> {
  const models = await listModels();
  return models.map((model) => ({
    name: model.name,
    size: model.size,
    vramUsed: 0,
  }));
}

export function setDefaultKeepAlive(keepAlive?: KeepAliveValue): void {
  // No-op for llama.cpp today (kept for runtime interface parity).
  defaultKeepAlive = keepAlive;
  void defaultKeepAlive;
}

export async function generate(
  model: string,
  prompt: string,
  options?: LlamaCppRequestOptions
): Promise<GenerateResult> {
  const start = Date.now();
  const controller = new AbortController();
  activeAbortControllers.add(controller);
  try {
    const resp = await negotiateChatRequest("generate", model, getLlamaCppBaseUrl(), options, (includeExtras) =>
      fetchLlamaCpp(
        "/v1/chat/completions",
        {
          method: "POST",
          headers: getLlamaCppHeaders(),
          body: JSON.stringify(buildChatBody(model, prompt, options, includeExtras, false)),
          signal: controller.signal,
        },
        LLAMA_CPP_INIT_TIMEOUT_MS,
        "llama.cpp generate request"
      )
    );

    const payload = (await resp.json()) as LlamaCppChatCompletionResponse;
    const choice = payload.choices?.[0];
    const response = asNonEmptyString(choice?.message?.content) ?? "";
    const reasoning = asNonEmptyString(choice?.message?.reasoning_content) ?? "";
    assertThinkingModeRespected(model, options?.think, response, reasoning);
    const usage = payload.usage;
    const timings = payload.timings;
    const totalDuration = Math.max(0, Date.now() - start) * 1e6;
    const outputTokens =
      getUsageTokenCount(usage?.completion_tokens)
      || getUsageTokenCount(timings?.predicted_n)
      || resolveCompletionTokenCount(undefined, response, reasoning);
    const promptTokens =
      getUsageTokenCount(usage?.prompt_tokens) || getUsageTokenCount(timings?.prompt_n);
    const promptEvalDuration =
      typeof timings?.prompt_ms === "number" && Number.isFinite(timings.prompt_ms)
        ? Math.max(0, Math.round(timings.prompt_ms * 1e6))
        : 0;
    const throughput = getUsageTokenCount(timings?.predicted_per_second);
    const evalDuration =
      typeof timings?.predicted_ms === "number" && Number.isFinite(timings.predicted_ms) && timings.predicted_ms > 0
        ? Math.round(timings.predicted_ms * 1e6)
        : throughput !== undefined && throughput > 0 && outputTokens > 0
          ? Math.max(1, Math.round((outputTokens / throughput) * 1e9))
          : totalDuration;
    const evalCountEstimated =
      getUsageTokenCount(usage?.completion_tokens) <= 0 && getUsageTokenCount(timings?.predicted_n) <= 0;

    return {
      response,
      ...(reasoning ? { thinking: reasoning } : {}),
      totalDuration,
      loadDuration: 0,
      promptEvalCount: promptTokens,
      promptEvalDuration,
      evalCount: outputTokens,
      evalDuration: Math.max(1, evalDuration),
      ...(evalCountEstimated ? { evalCountEstimated: true } : {}),
    };
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      throw new Error("llama.cpp generate request aborted");
    }
    throw err;
  } finally {
    activeAbortControllers.delete(controller);
  }
}

export async function generateStream(
  model: string,
  prompt: string,
  callbacks?: StreamCallbacks,
  options?: LlamaCppRequestOptions
): Promise<GenerateResult> {
  const start = Date.now();
  const controller = new AbortController();
  activeAbortControllers.add(controller);
  const stallTimeoutMs = resolveStreamStallTimeoutMs(options?.stall_timeout_ms);
  let abortedByStallTimeout = false;

  let stallTimer: ReturnType<typeof setTimeout> | null = null;
  const resetStallTimer = () => {
    if (stallTimeoutMs === undefined) return;
    if (stallTimer) clearTimeout(stallTimer);
    stallTimer = setTimeout(() => {
      abortedByStallTimeout = true;
      controller.abort();
    }, stallTimeoutMs);
  };

  try {
    resetStallTimer();

    const resp = await negotiateChatRequest("stream", model, getLlamaCppBaseUrl(), options, (includeExtras) =>
      fetchLlamaCpp(
        "/v1/chat/completions",
        {
          method: "POST",
          headers: getLlamaCppHeaders(),
          body: JSON.stringify(buildChatBody(model, prompt, options, includeExtras, true)),
          signal: controller.signal,
        },
        LLAMA_CPP_INIT_TIMEOUT_MS,
        "llama.cpp stream request"
      )
    );

    if (!resp.body) {
      throw new Error("llama.cpp stream response body is empty");
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";
    let doneReceived = false;
    let fullResponse = "";
    let fullThinking = "";
    let usage: LlamaCppChatCompletionResponse["usage"];
    let timings: LlamaCppChatCompletionResponse["timings"];
    let firstChunkSeen = false;
    let firstGeneratedTokenTime: number | null = null;
    let lastGeneratedTokenTime: number | null = null;

    const processChunk = (payload: LlamaCppChatCompletionChunk): void => {
      const delta = payload.choices?.[0]?.delta;
      const content = asNonEmptyString(delta?.content) ?? "";
      const reasoning = asNonEmptyString(delta?.reasoning_content) ?? "";
      if (payload.usage) usage = payload.usage;
      if (payload.timings) timings = payload.timings;

      if (reasoning || content) {
        const now = Date.now();
        if (firstGeneratedTokenTime === null) firstGeneratedTokenTime = now;
        lastGeneratedTokenTime = now;
      }
      if (reasoning) {
        fullThinking += reasoning;
      }
      if (content) {
        fullResponse += content;
        callbacks?.onToken?.(content);
      }
    };

    const processDataLine = (rawLine: string): void => {
      const line = rawLine.trim();
      if (!line.startsWith("data:")) return;
      const dataStr = line.slice(5).trim();
      if (!dataStr) return;
      if (dataStr === "[DONE]") {
        doneReceived = true;
        return;
      }

      let payload: LlamaCppChatCompletionChunk;
      try {
        payload = JSON.parse(dataStr) as LlamaCppChatCompletionChunk;
      } catch {
        return;
      }
      processChunk(payload);
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      resetStallTimer();
      if (!firstChunkSeen) {
        firstChunkSeen = true;
        callbacks?.onFirstChunk?.();
      }
      buffered += decoder.decode(value, { stream: true });
      const lines = buffered.split("\n");
      buffered = lines.pop() ?? "";
      for (const rawLine of lines) {
        processDataLine(rawLine);
      }
    }

    // Flush any pending decoder state for chunk boundaries / multibyte codepoints.
    buffered += decoder.decode();
    if (buffered.length > 0) {
      const lines = buffered.split("\n");
      buffered = lines.pop() ?? "";
      for (const rawLine of lines) {
        processDataLine(rawLine);
      }
    }

    if (buffered.trim().length > 0) {
      processDataLine(buffered);
    }

    if (stallTimer) clearTimeout(stallTimer);

    if (!doneReceived && !fullResponse && !fullThinking) {
      throw new Error("llama.cpp stream ended without content");
    }

    const totalDuration = Math.max(0, Date.now() - start) * 1e6;
    const outputTokens =
      getUsageTokenCount(usage?.completion_tokens)
      || getUsageTokenCount(timings?.predicted_n)
      || resolveCompletionTokenCount(undefined, fullResponse, fullThinking);
    const promptTokens =
      getUsageTokenCount(usage?.prompt_tokens) || getUsageTokenCount(timings?.prompt_n);
    const evalCountEstimated =
      getUsageTokenCount(usage?.completion_tokens) <= 0 && getUsageTokenCount(timings?.predicted_n) <= 0;
    // Prefer llama.cpp's native timing stats when available; otherwise fall
    // back to the token window measured from the streamed delta events.
    const evalDurationMs =
      typeof timings?.predicted_ms === "number" && Number.isFinite(timings.predicted_ms) && timings.predicted_ms > 0
        ? timings.predicted_ms
        : timings?.predicted_per_second !== undefined
          && timings.predicted_per_second > 0
          && outputTokens > 0
          ? (outputTokens / timings.predicted_per_second) * 1000
          : firstGeneratedTokenTime !== null
            && lastGeneratedTokenTime !== null
            && lastGeneratedTokenTime > firstGeneratedTokenTime
            ? lastGeneratedTokenTime - firstGeneratedTokenTime
            : Date.now() - start;
    const result: GenerateResult = {
      response: fullResponse,
      ...(fullThinking ? { thinking: fullThinking } : {}),
      totalDuration,
      loadDuration: 0,
      promptEvalCount: promptTokens,
      promptEvalDuration:
        typeof timings?.prompt_ms === "number" && Number.isFinite(timings.prompt_ms)
          ? Math.max(0, Math.round(timings.prompt_ms * 1e6))
          : firstGeneratedTokenTime !== null
            ? (firstGeneratedTokenTime - start) * 1e6
            : 0,
      evalCount: outputTokens,
      evalDuration: Math.max(1, Math.round(evalDurationMs * 1e6)),
      ...(evalCountEstimated ? { evalCountEstimated: true } : {}),
    };
    assertThinkingModeRespected(model, options?.think, fullResponse, fullThinking);

    callbacks?.onDone?.(result);
    return result;
  } catch (err) {
    if (err instanceof Error && err.name === "AbortError") {
      if (abortedByStallTimeout && stallTimeoutMs !== undefined) {
        throw new Error(`llama.cpp stream timed out after ${stallTimeoutMs}ms`);
      }
      throw new Error("llama.cpp stream request aborted");
    }
    throw err;
  } finally {
    if (stallTimer) clearTimeout(stallTimer);
    activeAbortControllers.delete(controller);
  }
}

export async function unloadModel(model: string): Promise<void> {
  // Only router-mode llama-server instances expose /models/unload.
  // Regular servers return 404/405, which is a no-op here.
  try {
    const resp = await fetchLlamaCpp(
      "/models/unload",
      {
        method: "POST",
        headers: getLlamaCppHeaders(),
        body: JSON.stringify({ model }),
      },
      LLAMA_CPP_METADATA_TIMEOUT_MS,
      "llama.cpp unload model"
    );
    if (resp.ok) {
      await resp.text().catch(() => undefined);
    }
  } catch {
    // Unloading is best-effort (e.g. non-router servers reject the endpoint).
  }
}

export function abortOngoingRequests(): void {
  for (const controller of activeAbortControllers) {
    controller.abort();
  }
  activeAbortControllers.clear();
}
