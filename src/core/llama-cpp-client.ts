import type { OllamaModel, OllamaRunningModel } from "../types.js";
import type { GenerateResult, KeepAliveValue, StreamCallbacks } from "./ollama-client.js";
import { estimateTokenCount } from "../utils.js";

const DEFAULT_LLAMA_CPP_BASE_URL = "http://127.0.0.1:8080";
const LLAMA_CPP_INIT_TIMEOUT_MS = 15_000;
const LLAMA_CPP_METADATA_TIMEOUT_MS = 5_000;
const DEFAULT_STREAM_STALL_TIMEOUT_MS = 30_000;
const SHARED_STREAM_STALL_TIMEOUT_ENV = "METRILLM_STREAM_STALL_TIMEOUT_MS";
const UNSUPPORTED_FIELD_PATTERN = /unrecognized|unknown|not support|unsupported|invalid|unexpected|additional|extra/i;

let defaultKeepAlive: KeepAliveValue | undefined;
const activeAbortControllers = new Set<AbortController>();

// ── API response types ─────────────────────────────────────

interface LlamaCppModelMeta {
  n_params?: number;
  size?: number;
  n_ctx_train?: number;
  n_embd?: number;
}

interface LlamaCppV1Model {
  id?: string;
  object?: string;
  created?: number;
  owned_by?: string;
  meta?: LlamaCppModelMeta | null;
}

interface LlamaCppV1ModelListResponse {
  object?: string;
  data?: LlamaCppV1Model[];
}

interface LlamaCppRouterModelStatus {
  value?: string;
}

interface LlamaCppRouterModel {
  id?: string;
  path?: string;
  status?: LlamaCppRouterModelStatus | null;
  meta?: LlamaCppModelMeta | null;
}

interface LlamaCppRouterModelListResponse {
  data?: LlamaCppRouterModel[];
}

interface LlamaCppTimings {
  cache_n?: number;
  prompt_n?: number;
  prompt_ms?: number;
  prompt_per_second?: number;
  predicted_n?: number;
  predicted_ms?: number;
  predicted_per_second?: number;
}

interface LlamaCppUsage {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
}

interface LlamaCppChatMessage {
  role?: string;
  content?: string;
  reasoning_content?: string;
}

interface LlamaCppChatChoice {
  index?: number;
  delta?: LlamaCppChatMessage;
  message?: LlamaCppChatMessage;
  finish_reason?: string | null;
}

interface LlamaCppChatCompletionPayload {
  id?: string;
  object?: string;
  choices?: LlamaCppChatChoice[];
  usage?: LlamaCppUsage | null;
  timings?: LlamaCppTimings | null;
}

interface LlamaCppRequestOptions {
  temperature?: number;
  top_p?: number;
  seed?: number;
  num_predict?: number;
  keep_alive?: KeepAliveValue;
  think?: boolean;
  stall_timeout_ms?: number;
}

// ── Connection helpers ─────────────────────────────────────

function getLlamaCppBaseUrl(): string {
  const configured = process.env.LLAMA_CPP_BASE_URL?.trim();
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
  const apiKey = process.env.LLAMA_CPP_API_KEY?.trim();
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }
  return headers;
}

async function fetchWithTimeout(
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

// ── Model metadata helpers ─────────────────────────────────

const KNOWN_QUANTIZATIONS = [
  "IQ1_S", "IQ1_M", "IQ1_L", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M", "IQ2_L",
  "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ3_L",
  "IQ4_XS", "IQ4_NL", "IQ4_S",
  "Q2_K", "Q3_K_S", "Q3_K_M", "Q3_K_L", "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M",
  "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q6_K", "Q8_0",
  "BF16", "F16", "F32",
] as const;

function inferQuantizationFromModelId(modelId: string): string | undefined {
  for (const token of KNOWN_QUANTIZATIONS) {
    const pattern = new RegExp(`(^|[^A-Z0-9_])${token}(?=$|[^A-Z0-9_])`, "i");
    if (pattern.test(modelId)) return token;
  }
  return undefined;
}

function normalizeModelNumber(value: string): string {
  return value.replace(/\.0+$/, "");
}

function inferParameterSizeFromModelId(modelId: string): string | undefined {
  const id = modelId.toLowerCase();
  const billionMatch = id.match(/\b(\d+(?:\.\d+)?)\s*b\b/);
  if (billionMatch?.[1]) {
    return `${normalizeModelNumber(billionMatch[1])}B`;
  }
  const millionMatch = id.match(/\b(\d+(?:\.\d+)?)\s*m\b/);
  if (millionMatch?.[1]) {
    return `${normalizeModelNumber(millionMatch[1])}M`;
  }
  return undefined;
}

function parameterSizeFromMeta(meta: LlamaCppModelMeta | null | undefined): string | undefined {
  const nParams = meta?.n_params;
  if (typeof nParams !== "number" || !Number.isFinite(nParams) || nParams <= 0) return undefined;
  const billions = nParams / 1e9;
  if (billions >= 1) {
    return `${normalizeModelNumber(billions.toFixed(1))}B`;
  }
  const millions = Math.round(nParams / 1e6);
  return `${millions}M`;
}

function modelSizeBytes(meta: LlamaCppModelMeta | null | undefined): number {
  const size = meta?.size;
  if (typeof size !== "number" || !Number.isFinite(size) || size <= 0) return 0;
  return Math.trunc(size);
}

function buildModelEntry(
  name: string,
  meta: LlamaCppModelMeta | null | undefined,
  status?: string
): OllamaModel {
  return {
    name,
    size: modelSizeBytes(meta),
    parameterSize: parameterSizeFromMeta(meta) ?? inferParameterSizeFromModelId(name),
    quantization: inferQuantizationFromModelId(name),
    modelFormat: "gguf",
    ...(status ? { runtimeStatus: status } : {}),
  };
}

function isLoadedStatus(status: string | undefined): boolean {
  return status?.trim().toLowerCase() === "loaded";
}

// ── Version / discovery ────────────────────────────────────

export async function getLlamaCppVersion(): Promise<string> {
  try {
    const resp = await fetchWithTimeout(
      "/props",
      { method: "GET", headers: getLlamaCppHeaders() },
      LLAMA_CPP_METADATA_TIMEOUT_MS,
      "llama.cpp props"
    );
    if (!resp.ok) return "unknown";
    const data = (await resp.json()) as { build_info?: string };
    if (typeof data.build_info === "string" && data.build_info.trim().length > 0) {
      return data.build_info.trim();
    }
    return "unknown";
  } catch {
    return "unknown";
  }
}

async function fetchRouterModels(): Promise<LlamaCppRouterModel[] | null> {
  try {
    const resp = await fetchWithTimeout(
      "/models",
      { method: "GET", headers: getLlamaCppHeaders() },
      LLAMA_CPP_METADATA_TIMEOUT_MS,
      "llama.cpp router models"
    );
    if (!resp.ok) return null;
    const data = (await resp.json()) as LlamaCppRouterModelListResponse;
    if (!Array.isArray(data.data)) return null;
    return data.data;
  } catch {
    return null;
  }
}

export async function listModels(): Promise<OllamaModel[]> {
  const routerModels = await fetchRouterModels();
  if (routerModels) {
    return routerModels
      .map((m) => buildModelEntry(m.id ?? m.path ?? "", m.meta, m.status?.value))
      .filter((m) => m.name.trim().length > 0);
  }

  const resp = await fetchWithTimeout(
    "/v1/models",
    { method: "GET", headers: getLlamaCppHeaders() },
    LLAMA_CPP_INIT_TIMEOUT_MS,
    "llama.cpp list models"
  );
  if (!resp.ok) {
    throw new Error(`llama.cpp list models failed (${resp.status} ${resp.statusText})`);
  }
  const data = (await resp.json()) as LlamaCppV1ModelListResponse;
  return (data.data ?? [])
    .filter((m) => typeof m.id === "string" && m.id.trim().length > 0)
    .map((m) => buildModelEntry(m.id as string, m.meta));
}

export async function listRunningModels(): Promise<OllamaRunningModel[]> {
  const routerModels = await fetchRouterModels();
  if (routerModels) {
    return routerModels
      .filter((m) => isLoadedStatus(m.status?.value))
      .map((m) => ({ name: m.id ?? m.path ?? "", size: modelSizeBytes(m.meta), vramUsed: 0 }))
      .filter((m) => m.name.trim().length > 0);
  }

  // Single-model server: whatever /v1/models reports is the served (loaded) model.
  const models = await listModels();
  return models.map((m) => ({ name: m.name, size: m.size, vramUsed: 0 }));
}

// ── Request building / negotiation ─────────────────────────

function hasSamplingOverrides(options?: LlamaCppRequestOptions): boolean {
  return options?.top_p !== undefined || options?.seed !== undefined;
}

function buildChatBody(
  model: string,
  prompt: string,
  options: LlamaCppRequestOptions | undefined,
  stream: boolean,
  includeSampling: boolean,
  includeReasoningControl: boolean
): Record<string, unknown> {
  return {
    model,
    messages: [{ role: "user", content: prompt }],
    temperature: options?.temperature ?? 0,
    max_tokens: options?.num_predict ?? 512,
    stream,
    ...(includeSampling && options?.top_p !== undefined ? { top_p: options.top_p } : {}),
    ...(includeSampling && options?.seed !== undefined ? { seed: options.seed } : {}),
    ...(includeReasoningControl && options?.think === false ? { reasoning_effort: "none" } : {}),
    ...(stream ? { stream_options: { include_usage: true } } : {}),
  };
}

function isUnsupportedFieldMessage(status: number, body: string, fieldName: string): boolean {
  if (status !== 400 && status !== 422) return false;
  const lower = body.toLowerCase();
  return lower.includes(fieldName.toLowerCase()) && UNSUPPORTED_FIELD_PATTERN.test(lower);
}

async function postChatWithFallbacks(
  model: string,
  prompt: string,
  options: LlamaCppRequestOptions | undefined,
  stream: boolean,
  controller: AbortController
): Promise<Response> {
  let includeSampling = true;
  let includeReasoningControl = true;
  const maxAttempts = 4;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    const body = JSON.stringify(
      buildChatBody(model, prompt, options, stream, includeSampling, includeReasoningControl)
    );
    const resp = await fetchWithTimeout(
      "/v1/chat/completions",
      { method: "POST", headers: getLlamaCppHeaders(), body, signal: controller.signal },
      LLAMA_CPP_INIT_TIMEOUT_MS,
      "llama.cpp chat completion"
    );
    if (resp.ok) return resp;
    const bodyText = await resp.text().catch(() => "");
    if (includeSampling && hasSamplingOverrides(options) && isUnsupportedFieldMessage(resp.status, bodyText, "seed")) {
      includeSampling = false;
      continue;
    }
    if (includeSampling && hasSamplingOverrides(options) && isUnsupportedFieldMessage(resp.status, bodyText, "top_p")) {
      includeSampling = false;
      continue;
    }
    if (includeReasoningControl && options?.think === false && isUnsupportedFieldMessage(resp.status, bodyText, "reasoning_effort")) {
      includeReasoningControl = false;
      continue;
    }
    throw new Error(
      `llama.cpp request failed (${resp.status} ${resp.statusText})${bodyText ? `: ${bodyText.slice(0, 300)}` : ""}`
    );
  }
  throw new Error("llama.cpp request failed after option fallback retries");
}

// ── Result mapping ─────────────────────────────────────────

function positiveInt(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  if (value < 0) return undefined;
  return Math.trunc(value);
}

function positiveNumber(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return undefined;
  return value;
}

interface LlamaCppResultInputs {
  response: string;
  thinking: string;
  usage: LlamaCppUsage | null;
  timings: LlamaCppTimings | null;
  firstTokenTime: number | null;
  lastTokenTime: number | null;
  start: number;
}

function buildGenerateResult(inputs: LlamaCppResultInputs): GenerateResult {
  const { response, thinking, usage, timings, firstTokenTime, lastTokenTime, start } = inputs;
  const totalDuration = Math.max(0, Date.now() - start) * 1e6;

  const reportedCompletion = positiveInt(usage?.completion_tokens) ?? positiveInt(timings?.predicted_n);
  const evalCountEstimated = reportedCompletion === undefined;
  const evalCount =
    reportedCompletion ?? Math.max(1, estimateTokenCount(`${thinking} ${response}`.trim()));

  const promptFromTimings =
    (positiveInt(timings?.prompt_n) ?? 0) + (positiveInt(timings?.cache_n) ?? 0);
  const promptEvalCount = positiveInt(usage?.prompt_tokens) ?? promptFromTimings;

  const promptEvalDurationMs = positiveNumber(timings?.prompt_ms);
  const promptEvalDuration = Math.max(0, Math.round((promptEvalDurationMs ?? 0) * 1e6));

  const predictedMs = positiveNumber(timings?.predicted_ms);
  let evalDurationMs: number;
  if (predictedMs !== undefined && predictedMs > 0) {
    evalDurationMs = predictedMs;
  } else if (firstTokenTime !== null && lastTokenTime !== null && lastTokenTime > firstTokenTime) {
    evalDurationMs = lastTokenTime - firstTokenTime;
  } else {
    evalDurationMs = Date.now() - start;
  }

  return {
    response,
    ...(thinking ? { thinking } : {}),
    totalDuration,
    loadDuration: 0,
    promptEvalCount,
    promptEvalDuration,
    evalCount,
    evalDuration: Math.max(1, Math.round(evalDurationMs * 1e6)),
    ...(evalCountEstimated ? { evalCountEstimated: true } : {}),
  };
}

function hasThinkingLeakText(response: string): boolean {
  return (
    /^\s*(?:thinking|thought)\s+process\s*:/i.test(response)
    || /\[(?:\/)?THINK(?:ING)?\]/i.test(response)
  );
}

function assertNonThinkingModeRespected(
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
        "Restart llama-server with --reasoning off for this model, or benchmark it with --thinking.",
      ].join(" ")
    );
  }
}

// ── Generation ─────────────────────────────────────────────

export function setDefaultKeepAlive(keepAlive?: KeepAliveValue): void {
  // llama.cpp has no per-request keep_alive; the server's --sleep-idle-seconds
  // controls unloading. Stored for runtime interface parity.
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
    const resp = await postChatWithFallbacks(model, prompt, options, false, controller);
    const data = (await resp.json()) as LlamaCppChatCompletionPayload;
    const choice = Array.isArray(data.choices) ? data.choices[0] : undefined;
    const message = choice?.message;
    const response = typeof message?.content === "string" ? message.content : "";
    const reasoning = typeof message?.reasoning_content === "string" ? message.reasoning_content : "";
    const result = buildGenerateResult({
      response,
      thinking: reasoning,
      usage: data.usage ?? null,
      timings: data.timings ?? null,
      firstTokenTime: null,
      lastTokenTime: null,
      start,
    });
    assertNonThinkingModeRespected(model, options?.think, response, reasoning);
    return result;
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
    const resp = await postChatWithFallbacks(model, prompt, options, true, controller);
    if (!resp.body) {
      throw new Error("llama.cpp stream response body is empty");
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffered = "";
    let fullResponse = "";
    let fullThinking = "";
    let usage: LlamaCppUsage | null = null;
    let timings: LlamaCppTimings | null = null;
    let firstChunkSeen = false;
    let firstTokenTime: number | null = null;
    let lastTokenTime: number | null = null;

    const processLine = (rawLine: string): void => {
      const line = rawLine.trim();
      if (!line.startsWith("data:")) return;
      const dataStr = line.slice(5).trim();
      if (!dataStr || dataStr === "[DONE]") return;
      let chunk: LlamaCppChatCompletionPayload;
      try {
        chunk = JSON.parse(dataStr) as LlamaCppChatCompletionPayload;
      } catch {
        return;
      }
      const choice = Array.isArray(chunk.choices) ? chunk.choices[0] : undefined;
      const delta = choice?.delta;
      const content = typeof delta?.content === "string" ? delta.content : "";
      const reasoning = typeof delta?.reasoning_content === "string" ? delta.reasoning_content : "";
      if (content || reasoning) {
        const now = Date.now();
        if (firstTokenTime === null) firstTokenTime = now;
        lastTokenTime = now;
      }
      if (reasoning) fullThinking += reasoning;
      if (content) {
        fullResponse += content;
        callbacks?.onToken?.(content);
      }
      if (chunk.usage) usage = chunk.usage;
      if (chunk.timings) timings = chunk.timings;
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
        processLine(rawLine);
      }
    }

    // Flush any pending decoder state for chunk boundaries / multibyte codepoints.
    buffered += decoder.decode();
    if (buffered.trim().length > 0) {
      processLine(buffered);
    }

    if (stallTimer) clearTimeout(stallTimer);

    const result = buildGenerateResult({
      response: fullResponse,
      thinking: fullThinking,
      usage,
      timings,
      firstTokenTime,
      lastTokenTime,
      start,
    });
    assertNonThinkingModeRespected(model, options?.think, fullResponse, fullThinking);

    callbacks?.onDone?.(result);
    return result;
  } catch (err) {
    if (stallTimer) clearTimeout(stallTimer);
    if (err instanceof Error && err.name === "AbortError") {
      if (abortedByStallTimeout && stallTimeoutMs !== undefined) {
        throw new Error(`llama.cpp stream timed out after ${stallTimeoutMs}ms`);
      }
      throw new Error("llama.cpp stream request aborted");
    }
    throw err;
  } finally {
    activeAbortControllers.delete(controller);
  }
}

// ── Model management ───────────────────────────────────────

export async function unloadModel(model: string): Promise<void> {
  const resp = await fetchWithTimeout(
    "/models/unload",
    {
      method: "POST",
      headers: getLlamaCppHeaders(),
      body: JSON.stringify({ model }),
    },
    LLAMA_CPP_INIT_TIMEOUT_MS,
    "llama.cpp unload model"
  );
  // Single-model servers have no /models/unload endpoint (router mode only).
  if (resp.status === 404 || resp.status === 405) return;
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(
      `llama.cpp unload failed (${resp.status} ${resp.statusText})${body ? `: ${body.slice(0, 300)}` : ""}`
    );
  }
}

export function abortOngoingRequests(): void {
  for (const controller of activeAbortControllers) {
    controller.abort();
  }
  activeAbortControllers.clear();
}
