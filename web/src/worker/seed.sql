INSERT OR REPLACE INTO benchmark_config (id, model, tasks, steps_per_task, lora_rank, seeds, memory)
VALUES (1, 'HuggingFaceTB/SmolLM2-135M-Instruct', '["math","code","ifeval","safety","domain"]', 1000, 64, '[42,137,256]', 128);

INSERT OR REPLACE INTO entries (method, description, contributor, date, score_mean, score_std, retention_mean, retention_std, plasticity_mean, plasticity_std, pr)
VALUES
  ('sfp', 'Sparse Feature Preservation — our method', 'paradigm', '2026-02-28T12:00:00Z', 0.7124, 0.0189, 0.754, 0.0231, 0.65, 0.0142, 'https://github.com/paradigmxyz/sfp/pull/1'),
  ('distill', 'LwF-style logit distillation + replay', 'alice', '2026-03-01T09:00:00Z', 0.635, 0.0198, 0.65, 0.0267, 0.6125, 0.013, ''),
  ('hidden_distill', 'POD-style hidden state matching + replay', 'carol', '2026-03-02T11:00:00Z', 0.618, 0.022, 0.63, 0.029, 0.6, 0.015, ''),
  ('strong_replay', 'Aggressive replay with 50% ratio', 'dave', '2026-03-02T16:30:00Z', 0.602, 0.026, 0.61, 0.03, 0.59, 0.022, ''),
  ('replay', 'Experience replay with 128-token memory buffer', 'paradigm', '2026-02-27T10:00:00Z', 0.5892, 0.0245, 0.582, 0.0312, 0.6, 0.0178, ''),
  ('orthogonal', 'Orthogonal LoRA constraint on adapter weights', 'bob', '2026-03-01T14:00:00Z', 0.551, 0.0275, 0.535, 0.034, 0.575, 0.021, ''),
  ('sfp_random_basis', 'SFP with random orthonormal basis (ablation)', 'paradigm', '2026-03-02T15:00:00Z', 0.505, 0.033, 0.47, 0.041, 0.5575, 0.025, ''),
  ('naive', 'Standard fine-tuning, no forgetting mitigation', 'paradigm', '2026-02-26T08:00:00Z', 0.328, 0.031, 0.18, 0.042, 0.55, 0.02, '');
