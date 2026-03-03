import { useEffect, useState } from "react";
import type { Benchmark, Entry } from "@/lib/types";
import { fetchLeaderboard } from "@/lib/api";
import LeaderboardTable from "@/components/LeaderboardTable";
import ScatterPlot from "@/components/ScatterPlot";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";

export default function Home() {
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    async function load() {
      const lb = await fetchLeaderboard();
      if (lb) {
        setBenchmark(lb.benchmark);
        setEntries(lb.entries);
      } else {
        setError(true);
      }
      setLoading(false);
    }
    load();
  }, []);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center" style={{ color: "var(--text-dim)" }}>
        Loading...
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center" style={{ color: "var(--text-dim)" }}>
        Failed to load leaderboard data.
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <Nav />

      {/* ── Intro ── */}
      <section className="mb-4 text-sm leading-relaxed" style={{ color: "var(--text-dim)" }}>
        <ul className="list-inside list-disc space-y-0.5">
          <li>Fine-tune an LLM on tasks A, B, C in sequence — <strong style={{ color: "var(--text)" }}>how much of A does it forget?</strong></li>
          <li>Every method below is a single loss function. Same model, same data, same compute.</li>
          <li>The only variable is your training strategy. <a href="/about" style={{ color: "var(--accent)" }}>Read the research question →</a></li>
          <li>Write a loss function and <a href="/submit" style={{ color: "var(--accent)" }}>submit it</a> — we run it on a cloud GPU and score it automatically.</li>
        </ul>
      </section>

      {/* ── Leaderboard + Scatter side by side ── */}
      <section className="mb-4 grid items-start gap-6 lg:grid-cols-2">
        <div>
          <div className="mb-2">
            <h2 className="text-lg font-semibold">Leaderboard</h2>
            <span className="text-xs" style={{ color: "var(--text-dim)" }}>
              Score = 0.6 × retention + 0.4 × plasticity
            </span>
          </div>
          <LeaderboardTable entries={entries} />
        </div>
        <div>
          <ScatterPlot entries={entries} />
        </div>
      </section>

      {/* ── Benchmark config + Submit CTA ── */}
      <section className="mb-4 grid items-stretch gap-6 lg:grid-cols-2">
        {benchmark && (
          <div
            className="grid grid-cols-2 gap-x-6 gap-y-1 rounded-md p-4 text-xs sm:grid-cols-3"
            style={{ border: "1px solid var(--border)", background: "var(--surface)" }}
          >
            {[
              ["Model", benchmark.model],
              ["Tasks", benchmark.tasks.join(" → ")],
              ["Steps/task", benchmark.steps_per_task.toLocaleString()],
              ["LoRA rank", benchmark.lora_rank],
              ["Seeds", benchmark.seeds.join(", ")],
              ["Memory", `${benchmark.memory} tokens`],
            ].map(([label, value]) => (
              <div key={label as string}>
                <span style={{ color: "var(--text-dim)" }}>{label as string}: </span>
                <span className="font-semibold">{value as string}</span>
              </div>
            ))}
          </div>
        )}
        <div
          className="flex items-center justify-between rounded-md p-4"
          style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
        >
          <div>
            <div className="text-sm font-semibold" style={{ color: "var(--text)" }}>
              Think you can do better?
            </div>
            <div className="text-xs" style={{ color: "var(--text-dim)" }}>
              Write a loss function, submit it, get scored. No GPU needed.
            </div>
          </div>
          <a
            href="/submit"
            className="rounded-md px-4 py-2 text-sm font-semibold"
            style={{ background: "var(--accent)", color: "#000" }}
          >
            Submit →
          </a>
        </div>
      </section>

      <Footer />
    </div>
  );
}
