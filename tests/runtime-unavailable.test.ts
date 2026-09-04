import { describe, expect, it } from "vitest";
import { getRuntimeUnavailableHelp } from "../src/core/runtime-unavailable.js";

describe("getRuntimeUnavailableHelp", () => {
  it("includes backend-selection guidance for Ollama", () => {
    expect(
      getRuntimeUnavailableHelp("ollama", [
        "Start it with:  ollama serve",
        "Install it at:  https://ollama.com",
      ])
    ).toEqual([
      "MetriLLM is currently set to use Ollama.",
      "Either start Ollama, or switch to another backend in Settings.",
      "  • Start it with:  ollama serve",
      "  • Install it at:  https://ollama.com",
      "  • To change backend: Main Menu -> Settings -> Runtime backend",
    ]);
  });

  it("includes backend-selection guidance for LM Studio", () => {
    expect(
      getRuntimeUnavailableHelp("lm-studio", [
        "Start LM Studio local server (Developer tab -> Local Server).",
      ])
    ).toEqual([
      "MetriLLM is currently set to use LM Studio.",
      "Either start LM Studio, or switch to another backend in Settings.",
      "  • Start LM Studio local server (Developer tab -> Local Server).",
      "  • To change backend: Main Menu -> Settings -> Runtime backend",
    ]);
  });

  it("includes backend-selection guidance for llama.cpp", () => {
    expect(
      getRuntimeUnavailableHelp("llama-cpp", [
        "Start llama-server with a GGUF model: llama-server -m <model>.gguf -c 8192",
        "Optionally set LLAMA_CPP_BASE_URL if your server is not on http://127.0.0.1:8080.",
      ])
    ).toEqual([
      "MetriLLM is currently set to use llama.cpp.",
      "Either start llama.cpp, or switch to another backend in Settings.",
      "  • Start llama-server with a GGUF model: llama-server -m <model>.gguf -c 8192",
      "  • Optionally set LLAMA_CPP_BASE_URL if your server is not on http://127.0.0.1:8080.",
      "  • To change backend: Main Menu -> Settings -> Runtime backend",
    ]);
  });
});
