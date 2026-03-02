import { useEffect, useState } from "react";
import type { Benchmark, Entry, DetailEntry } from "@/lib/types";
import { fetchLeaderboard, fetchDetails } from "@/lib/api";
import ForgettingChart from "@/components/ForgettingChart";
import LeaderboardTable from "@/components/LeaderboardTable";
import ScatterPlot from "@/components/ScatterPlot";
import BenchmarkInfo from "@/components/BenchmarkInfo";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";

export default function Home() {
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [details, setDetails] = useState<DetailEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    async function load() {
      const [lb, det] = await Promise.all([fetchLeaderboard(), fetchDetails()]);
      if (lb) {
        setBenchmark(lb.benchmark);
        setEntries(lb.entries);
      }
      if (det) {
        setDetails(det.entries);
      }
      if (!lb && !det) setError(true);
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
    <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
      <Nav />

      {/* ── Leaderboard (hero) ── */}
      <section className="mb-10">
        <div className="mb-3 flex items-baseline justify-between">
          <h2 className="text-xl font-semibold">Leaderboard</h2>
          <span className="text-xs" style={{ color: "var(--text-dim)" }}>
            Score = 0.6 × retention + 0.4 × plasticity
          </span>
        </div>
        <LeaderboardTable entries={entries} />
      </section>

      {/* ── Visualizations ── */}
      <section className="mb-10 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide" style={{ color: "var(--text-dim)" }}>
            Retention vs Plasticity
          </h3>
          <ScatterPlot entries={entries} />
        </div>
        <div>
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide" style={{ color: "var(--text-dim)" }}>
            Forgetting Over Training
          </h3>
          <ForgettingChart details={details} tasks={benchmark?.tasks ?? []} />
        </div>
      </section>

      {/* ── Submit CTA ── */}
      <section
        className="mb-10 flex items-center justify-between rounded-md p-4"
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
      </section>

      {/* ── Benchmark Details ── */}
      {benchmark && (
        <section className="mb-10">
          <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide" style={{ color: "var(--text-dim)" }}>
            Benchmark Configuration
          </h3>
          <BenchmarkInfo benchmark={benchmark} />
        </section>
      )}

      {/* ── Metrics ── */}
      <section className="mb-10">
        <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide" style={{ color: "var(--text-dim)" }}>
          Metrics
        </h3>
        <div className="grid gap-4 sm:grid-cols-3">
          {[
            {
              name: "Retention",
              desc: "Mean accuracy on previously-learned tasks. Measures how well your method prevents catastrophic forgetting.",
            },
            {
              name: "Plasticity",
              desc: "Accuracy on the most recently trained task. Measures how well the model can still learn new things.",
            },
            {
              name: "Score",
              desc: "0.6 × retention + 0.4 × plasticity. Weighted composite favoring retention — naive fine-tuning already achieves high plasticity.",
            },
          ].map((m) => (
            <div key={m.name} className="rounded-md p-3" style={{ border: "1px solid var(--border)" }}>
              <div className="mb-1 text-sm font-semibold" style={{ color: "var(--text)" }}>{m.name}</div>
              <div className="text-xs leading-relaxed" style={{ color: "var(--text-dim)" }}>{m.desc}</div>
            </div>
          ))}
        </div>
      </section>

      <Footer />
    </div>
  );
}
