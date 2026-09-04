export type Verdict = "EXCELLENT" | "GOOD" | "MARGINAL" | "NOT RECOMMENDED";

export interface HardwareInfo {
  os: string;
  arch: string;
  cpu: string;
  cpuCores: number;
  totalMemoryGB: number;
  gpu: string;
  vramGB: number;
  powerMode: "plugged-in" | "battery" | "low-power";
  machineModel?: string; // e.g. "Mac16,6"
  batteryPercent?: number | null;
  cpuFreqGHz?: number | null;
  cpuCurrentSpeedGHz?: number | null;
}

export interface OllamaModel {
  name: string;
  size: number;
  parameterSize?: string;
  quantization?: string;
  family?: string;
  modelFormat?: string;
  runtimeStatus?: string;
}

export interface OllamaRunningModel {
  name: string;
  size: number;
  vramUsed: number;
}

export interface PerformanceMetrics {
  tokensPerSecond: number;
  tokensPerSecondEstimated?: boolean;
  firstChunkMs?: number;
  ttft: number;
  loadTime: number;
  loadTimeAvailable: boolean;
  totalTokens: number;
  promptTokens: number;
  completionTokens: number;
  memoryUsedGB: number;
  memoryPercent: number;
  memoryFootprintAvailable: boolean;
  memoryFootprintEstimated?: boolean;
  memoryHostUsedGB: number;
  memoryHostPercent: number;
  tpsStdDev?: number;
  thinkingTokensEstimate?: number;
}

export interface BenchEnvironment {
  thermalPressureBefore: "nominal" | "fair" | "serious" | "severe" | "critical" | "unknown";
  thermalPressureAfter: "nominal" | "fair" | "serious" | "severe" | "critical" | "unknown";
  swapDeltaGB?: number;
  batteryPowered?: boolean;
  cpuAvgLoad?: number;
  cpuPeakLoad?: number;
}

export interface ModelInfo {
  parameterSize?: string;
  quantization?: string;
  family?: string;
  thinkingDetected?: boolean;
}

export interface SubmitterInfo {
  nickname: string;
  emailHash: string;
}

export interface BenchmarkProfile {
  version: string;
  temperature: number;
  topP: number;
  seed: number;
  thinkingMode: boolean;
  contextWindow: "runtime-default";
}

export interface RunMetadata {
  benchmarkSpecVersion: string;
  promptPackVersion: string;
  runtimeVersion: string;
  runtimeBackend: "ollama" | "lm-studio" | "llama-cpp";
  modelFormat: string; // e.g. "gguf", "mlx", "safetensors"
  benchmarkProfile: BenchmarkProfile;
  rawLogHash: string;
}

export interface BenchResult {
  model: string;
  modelInfo: ModelInfo;
  hardware: HardwareInfo;
  performance: PerformanceMetrics;
  quality: QualityMetrics | null;
  fitness: FitnessResult;
  benchEnvironment?: BenchEnvironment;
  submitter?: SubmitterInfo;
  timestamp: string;
  metadata: RunMetadata;
}

export interface QualityMetrics {
  reasoning: { score: number; max: number };
  math: { score: number; max: number };
  coding: { score: number; max: number };
  instructionFollowing: { score: number; max: number };
  structuredOutput: { score: number; max: number };
  multilingual: { score: number; max: number };
}

export interface ScoreBreakdown {
  speed: number;
  ttft: number;
  memory: number;
  total: number;
}

export interface TimePenalties {
  ttftPenaltyPct?: number;
  codingPenaltyPct?: number;
}

export interface QualityScore {
  total: number;
  reasoning: number;
  math: number;
  coding: number;
  instructionFollowing: number;
  structuredOutput: number;
  multilingual: number;
  timePenalties?: TimePenalties;
}

export interface FitnessResult {
  hardwareFitScore: number;
  performanceScore: ScoreBreakdown;
  qualityScore: QualityScore | null;
  globalScore: number | null; // null when perf-only
  verdict: Verdict;
  interpretation: string;
  warnings: string[];
  tierLabel: string;
}

export type BenchScope = "full" | "perf";
