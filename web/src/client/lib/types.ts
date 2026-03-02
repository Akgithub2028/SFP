export interface Entry {
  method: string;
  description: string;
  contributor: string;
  date: string;
  score_mean: number;
  score_std: number;
  retention_mean: number;
  retention_std: number;
  plasticity_mean: number;
  plasticity_std: number;
  pr?: string;
}

export interface Benchmark {
  model: string;
  tasks: string[];
  steps_per_task: number;
  lora_rank: number;
  seeds: number[];
  memory: number;
}

export interface LeaderboardData {
  benchmark: Benchmark;
  entries: Entry[];
}

export interface PerSeed {
  history: Record<string, Record<string, number>>[];
  retention: number;
  plasticity: number;
  score: number;
}

export interface DetailEntry {
  method: string;
  score_mean: number;
  per_seed: PerSeed[];
}

export interface DetailData {
  entries: DetailEntry[];
}


