# Multi-Backend Architecture

MetriLLM supports multiple LLM inference backends through the `LLMRuntime` interface. Two dimensions affect benchmark scores:

- **Runtime backend** — the inference engine (Ollama, LM Studio, mlx-lm, llama.cpp, vLLM)
- **Model format** — the weight format reported by the runtime (for example: GGUF, MLX, Safetensors, ONNX, GGML)

These are independent: a single backend (e.g. LM Studio) can serve both GGUF and MLX models.

## Supported Backends

| Backend    | Format(s)          | API                  | Default Port | Detection            | Status |
|------------|--------------------|----------------------|--------------|----------------------|--------|
| Ollama     | GGUF               | REST `/api`          | 11434        | `ollama serve`       | Stable |
| LM Studio  | Runtime-dependent  | Native REST          | 1234         | `/api/v1/models`     | Stable |
| llama.cpp  | GGUF               | OpenAI-compat `/v1`  | 8080         | `/v1/models` + `/props` | Stable |
| mlx-lm     | MLX                | OpenAI-compat        | 8080         | `/v1/models`         | Planned |
| vLLM       | Safetensors, GGUF  | OpenAI-compat        | 8000         | `/v1/models`         | Planned |

LM Studio notes:
- MetriLLM now uses LM Studio's native REST API, including `/api/v1/chat` for inference and `/api/v1/models` for primary model discovery.
- The previous OpenAI-compatible inference endpoint `/v1/chat/completions` has been removed from the LM Studio runtime adapter.
- Model discovery still uses LM Studio model listing endpoints because they expose inventory/runtime metadata needed by the CLI.

llama.cpp notes:
- MetriLLM targets the `llama-server` HTTP API (OpenAI-compatible `/v1/chat/completions` for inference, `/v1/models` for single-model discovery, and the router-mode `GET /models` endpoint for multi-model servers).
- Version/build info comes from `GET /props` (`build_info` field) and degrades to `unknown` gracefully.
- Token metrics prefer `usage` (requested via `stream_options.include_usage`) and the server `timings` block; when those are absent, MetriLLM estimates the completion count and flags `tokensPerSecondEstimated`.
- Model unloading uses `POST /models/unload` (router mode). On single-model servers the endpoint does not exist (404/405) and is treated as a no-op.
- Non-thinking mode sends `reasoning_effort: "none"` and verifies that no reasoning content leaks; if the server rejects the field, it is dropped and the request is retried without it.
- Endpoint and auth are configurable via `LLAMA_CPP_BASE_URL` and `LLAMA_CPP_API_KEY`.

Shared stream stall timeout:
- MetriLLM uses one cross-backend stream watchdog flag: `--stream-stall-timeout-ms`.
- The matching environment variable is `METRILLM_STREAM_STALL_TIMEOUT_MS`.
- Default is `30000` ms for all backends; `0` disables the watchdog.

## Model Formats

Common examples MetriLLM may encounter:

| Format      | Extension      | Quantization    | Typical Use                  |
|-------------|----------------|-----------------|------------------------------|
| GGUF        | `.gguf`        | Q4_K_M, Q5_K_M | CPU + GPU offload (llama.cpp)|
| MLX         | `.safetensors` | 4-bit, 8-bit   | Apple Silicon native (MLX)   |
| Safetensors | `.safetensors` | FP16, BF16      | GPU inference (vLLM, TGI)    |
| ONNX        | `.onnx`        | INT8, FP16      | Cross-platform optimized     |
| GGML        | varies         | legacy / mixed  | Older llama-family runtimes  |

MetriLLM stores the exact runtime-reported format when available. If the backend cannot provide a trustworthy format, the result is stored as `unknown` rather than guessed.

## Architecture

### LLMRuntime Interface

```typescript
export interface LLMRuntime {
  name: string;              // "ollama" | "lm-studio" | "llama-cpp" | "mlx" | "vllm"
  modelFormat?: string;      // runtime default format hint (not the exact per-model saved format)
  generate(...): Promise<GenerateResult>;
  generateStream(...): Promise<GenerateResult>;
  listModels(): Promise<OllamaModel[]>;
  listRunningModels(): Promise<OllamaRunningModel[]>;
  getVersion(): Promise<string>;
  unloadModel(model: string): Promise<void>;
  setKeepAlive(keepAlive?: KeepAliveValue): void;
  abort(): void;
}
```

### Runtime Selection

```typescript
import { setRuntime, getRuntime, getRuntimeName, getRuntimeModelFormat } from "./core/runtime.js";

// Default: OllamaRuntime
// Switch backend:
setRuntimeByName("llama-cpp");

// Access backend info:
getRuntimeName();        // "llama-cpp"
getRuntimeModelFormat(); // runtime default hint, e.g. "gguf"
```

## Database Schema

Two columns in the `benchmarks` table store backend information:

```sql
runtime_backend text not null default 'ollama'   -- indexed
model_format    text not null default 'gguf'      -- indexed
```

These are populated from `RunMetadata.runtimeBackend` and `RunMetadata.modelFormat` during upload.

## Adding a New Backend — Checklist

1. **Create client module** — `src/core/<backend>-client.ts` for low-level API calls

2. **Register runtime class** — Add a class implementing `LLMRuntime` in `src/core/runtime.ts`
   - Set `name` to the backend identifier (e.g. `"llama-cpp"`)
   - Set `modelFormat` to the default format (e.g. `"gguf"`)
   - Implement all interface methods (generate, listModels, etc.)
   - Add the name to `SUPPORTED_RUNTIME_BACKENDS`, `RUNTIME_LABELS`, and the setup/install hint helpers

3. **Register in factory** — Add backend to `createRuntime` / `normalizeRuntimeBackend` in `src/core/runtime.ts`

4. **Update config parsing** — `src/core/store.ts` (`parseRuntimeBackend`) and `src/ui/menu.ts` (`resolveConfiguredBackend`, backend selector)

5. **Populate metadata** — `getRuntimeName()` is auto-populated via the runtime proxy; exact `modelFormat` should come from per-model runtime metadata when available

6. **Update MCP** — Add the runtime to `SUPPORTED_RUNTIMES` in `mcp/src/tools.ts` and extend `normalizeResultRuntimeBackend`

7. **Add tests** — Unit tests for the new client, runtime switching, MCP forwarding, and unavailable-runtime help

8. **Update types** — Add backend name to `RunMetadata.runtimeBackend` JSDoc union

9. **Update docs** — `README.md`, `docs/multi-backend.md`, `AGENTS.md`, `.env.example`

10. **Update companion site** — Mirror type changes in `metrillm-web`
