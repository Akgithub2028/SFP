CREATE TABLE IF NOT EXISTS benchmark_config (
  id INTEGER PRIMARY KEY DEFAULT 1,
  model TEXT NOT NULL,
  tasks TEXT NOT NULL,  -- JSON array
  steps_per_task INTEGER NOT NULL,
  lora_rank INTEGER NOT NULL,
  seeds TEXT NOT NULL,  -- JSON array
  memory INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS entries (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  method TEXT NOT NULL UNIQUE,
  description TEXT DEFAULT '',
  contributor TEXT DEFAULT '',
  date TEXT DEFAULT '',
  score_mean REAL NOT NULL DEFAULT 0,
  score_std REAL NOT NULL DEFAULT 0,
  retention_mean REAL NOT NULL DEFAULT 0,
  retention_std REAL NOT NULL DEFAULT 0,
  plasticity_mean REAL NOT NULL DEFAULT 0,
  plasticity_std REAL NOT NULL DEFAULT 0,
  pr TEXT DEFAULT '',
  per_seed TEXT DEFAULT '[]',  -- JSON array of PerSeed objects
  created_at TEXT DEFAULT (datetime('now'))
);
