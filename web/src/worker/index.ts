import { Hono } from "hono";
import { cors } from "hono/cors";

type Bindings = {
  DB: D1Database;
};

const api = new Hono<{ Bindings: Bindings }>();

api.use("/api/*", cors());

api.get("/api/leaderboard", async (c) => {
  const db = c.env.DB;
  const entries = await db
    .prepare(
      `SELECT method, description, contributor, date,
              score_mean, score_std,
              retention_mean, retention_std,
              plasticity_mean, plasticity_std,
              pr
       FROM entries ORDER BY score_mean DESC`
    )
    .all();

  const config = await db
    .prepare("SELECT * FROM benchmark_config LIMIT 1")
    .first();

  return c.json({
    benchmark: config
      ? {
          model: config.model as string,
          tasks: JSON.parse(config.tasks as string),
          steps_per_task: config.steps_per_task as number,
          lora_rank: config.lora_rank as number,
          seeds: JSON.parse(config.seeds as string),
          memory: config.memory as number,
        }
      : null,
    entries: entries.results ?? [],
  });
});

api.get("/api/leaderboard/details", async (c) => {
  const db = c.env.DB;
  const entries = await db
    .prepare(
      `SELECT method, score_mean, per_seed
       FROM entries ORDER BY score_mean DESC`
    )
    .all();

  return c.json({
    entries: (entries.results ?? []).map((e: Record<string, unknown>) => ({
      method: e.method,
      score_mean: e.score_mean,
      per_seed: e.per_seed ? JSON.parse(e.per_seed as string) : [],
    })),
  });
});

export default api;
