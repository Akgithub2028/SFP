INSERT OR REPLACE INTO benchmark_config (id, model, tasks, steps_per_task, lora_rank, seeds, memory)
VALUES (1, 'Qwen/Qwen2.5-1.5B-Instruct', '["math","code","ifeval","safety","domain"]', 2000, 64, '[42,43,44]', 128);

-- Scores below use harmonic mean of retention & plasticity.
-- Placeholder values — will be replaced when the benchmark runs.
INSERT OR REPLACE INTO entries (method, description, contributor, date, score_mean, score_std, retention_mean, retention_std, plasticity_mean, plasticity_std, pr)
VALUES
  ('sfp', 'Sparse Feature Preservation — our method', 'paradigm', '2026-02-28T12:00:00Z', 0.699, 0.019, 0.754, 0.023, 0.650, 0.014, 'https://github.com/paradigmxyz/sfp/pull/1'),
  ('distill', 'LwF-style logit distillation + replay', 'alice', '2026-03-01T09:00:00Z', 0.631, 0.020, 0.650, 0.027, 0.613, 0.013, ''),
  ('hidden_distill', 'POD-style hidden state matching + replay', 'carol', '2026-03-02T11:00:00Z', 0.615, 0.022, 0.630, 0.029, 0.600, 0.015, ''),
  ('strong_replay', 'Aggressive replay with 50% ratio', 'dave', '2026-03-02T16:30:00Z', 0.600, 0.026, 0.610, 0.030, 0.590, 0.022, ''),
  ('replay', 'Experience replay with 128-token memory buffer', 'paradigm', '2026-02-27T10:00:00Z', 0.591, 0.025, 0.582, 0.031, 0.600, 0.018, ''),
  ('orthogonal', 'Orthogonal LoRA constraint on adapter weights', 'bob', '2026-03-01T14:00:00Z', 0.554, 0.028, 0.535, 0.034, 0.575, 0.021, ''),
  ('sfp_random_basis', 'SFP with random orthonormal basis (ablation)', 'paradigm', '2026-03-02T15:00:00Z', 0.510, 0.033, 0.470, 0.041, 0.558, 0.025, ''),
  ('naive', 'Standard fine-tuning, no forgetting mitigation', 'paradigm', '2026-02-26T08:00:00Z', 0.000, 0.000, 0.180, 0.042, 0.550, 0.020, '');
