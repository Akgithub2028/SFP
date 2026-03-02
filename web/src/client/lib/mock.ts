/**
 * Mock data for local UI development.
 * Used when the backend API is unreachable.
 */
import type { LeaderboardData, DetailData } from "./types";

const TASKS = ["math", "code", "ifeval", "safety", "domain"];

export const MOCK_LEADERBOARD: LeaderboardData = {
  benchmark: {
    model: "HuggingFaceTB/SmolLM2-135M-Instruct",
    tasks: TASKS,
    steps_per_task: 1000,
    lora_rank: 64,
    seeds: [42, 137, 256],
    memory: 128,
  },
  entries: [
    {
      method: "sfp",
      description: "Sparse Feature Preservation — our method",
      contributor: "paradigm",
      date: "2026-02-28T12:00:00Z",
      score_mean: 0.7124,
      score_std: 0.0189,
      retention_mean: 0.7540,
      retention_std: 0.0231,
      plasticity_mean: 0.6500,
      plasticity_std: 0.0142,
      pr: "https://github.com/paradigmxyz/sfp/pull/1",
    },
    {
      method: "replay",
      description: "Experience replay with 128-token memory buffer",
      contributor: "paradigm",
      date: "2026-02-27T10:00:00Z",
      score_mean: 0.5892,
      score_std: 0.0245,
      retention_mean: 0.5820,
      retention_std: 0.0312,
      plasticity_mean: 0.6000,
      plasticity_std: 0.0178,
    },
    {
      method: "distill",
      description: "LwF-style logit distillation + replay",
      contributor: "alice",
      date: "2026-03-01T09:00:00Z",
      score_mean: 0.6350,
      score_std: 0.0198,
      retention_mean: 0.6500,
      retention_std: 0.0267,
      plasticity_mean: 0.6125,
      plasticity_std: 0.0130,
    },
    {
      method: "naive",
      description: "Standard fine-tuning, no forgetting mitigation",
      contributor: "paradigm",
      date: "2026-02-26T08:00:00Z",
      score_mean: 0.3280,
      score_std: 0.0310,
      retention_mean: 0.1800,
      retention_std: 0.0420,
      plasticity_mean: 0.5500,
      plasticity_std: 0.0200,
    },
    {
      method: "orthogonal",
      description: "Orthogonal LoRA constraint on adapter weights",
      contributor: "bob",
      date: "2026-03-01T14:00:00Z",
      score_mean: 0.5510,
      score_std: 0.0275,
      retention_mean: 0.5350,
      retention_std: 0.0340,
      plasticity_mean: 0.5750,
      plasticity_std: 0.0210,
    },
    {
      method: "hidden_distill",
      description: "POD-style hidden state matching + replay",
      contributor: "carol",
      date: "2026-03-02T11:00:00Z",
      score_mean: 0.6180,
      score_std: 0.0220,
      retention_mean: 0.6300,
      retention_std: 0.0290,
      plasticity_mean: 0.6000,
      plasticity_std: 0.0150,
    },
    {
      method: "sfp_random_basis",
      description: "SFP with random orthonormal basis (ablation)",
      contributor: "paradigm",
      date: "2026-03-02T15:00:00Z",
      score_mean: 0.5050,
      score_std: 0.0330,
      retention_mean: 0.4700,
      retention_std: 0.0410,
      plasticity_mean: 0.5575,
      plasticity_std: 0.0250,
    },
    {
      method: "strong_replay",
      description: "Aggressive replay with 50% ratio",
      contributor: "dave",
      date: "2026-03-02T16:30:00Z",
      score_mean: 0.6020,
      score_std: 0.0260,
      retention_mean: 0.6100,
      retention_std: 0.0300,
      plasticity_mean: 0.5900,
      plasticity_std: 0.0220,
    },
  ],
};

// Helper: generate plausible per-task accuracy curves
function makeHistory(
  tasks: string[],
  retentionFinal: number,
  plasticityFinal: number,
  noise: number = 0.03,
): Record<string, Record<string, number>>[] {
  const PRIMARY: Record<string, string> = {
    math: "math_accuracy",
    code: "code_pass_at_1",
    ifeval: "ifeval_accuracy",
    safety: "refusal_rate",
    domain: "domain_qa_accuracy",
  };

  const history: Record<string, Record<string, number>>[] = [];

  for (let stage = 0; stage < tasks.length; stage++) {
    const stageResult: Record<string, Record<string, number>> = {};

    for (let ti = 0; ti <= stage; ti++) {
      const task = tasks[ti];
      const metric = PRIMARY[task] ?? `${task}_accuracy`;

      let value: number;
      if (ti === stage) {
        // Just trained on this task — high plasticity
        value = plasticityFinal + (Math.random() - 0.5) * noise;
      } else {
        // Earlier task — decays based on how far back, modulated by retention
        const decay = (stage - ti) * (1 - retentionFinal) * 0.15;
        const baseAcc = 0.6 + retentionFinal * 0.3;
        value = baseAcc - decay + (Math.random() - 0.5) * noise;
      }
      stageResult[task] = { [metric]: Math.max(0, Math.min(1, value)) };
    }
    history.push(stageResult);
  }
  return history;
}

function makePerSeed(
  entry: typeof MOCK_LEADERBOARD.entries[0],
  tasks: string[],
  nSeeds: number = 3,
) {
  return Array.from({ length: nSeeds }, () => {
    const ret = entry.retention_mean + (Math.random() - 0.5) * entry.retention_std * 2;
    const plas = entry.plasticity_mean + (Math.random() - 0.5) * entry.plasticity_std * 2;
    return {
      history: makeHistory(tasks, ret, plas),
      retention: ret,
      plasticity: plas,
      score: 0.6 * ret + 0.4 * plas,
    };
  });
}

export const MOCK_DETAILS: DetailData = {
  entries: MOCK_LEADERBOARD.entries.map((e) => ({
    method: e.method,
    score_mean: e.score_mean,
    per_seed: makePerSeed(e, TASKS),
  })),
};
