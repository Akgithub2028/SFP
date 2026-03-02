"use client";

import { useEffect, useState } from "react";
import type { Benchmark, Entry, DetailEntry } from "@/lib/types";
import { fetchLeaderboard, fetchDetails } from "@/lib/api";
import ForgettingChart from "@/components/ForgettingChart";
import LeaderboardTable from "@/components/LeaderboardTable";
import ScatterPlot from "@/components/ScatterPlot";
import BenchmarkInfo from "@/components/BenchmarkInfo";
import PrometheusCharts from "@/components/PrometheusCharts";

export default function Home() {
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);
  const [entries, setEntries] = useState<Entry[]>([]);
  const [details, setDetails] = useState<DetailEntry[]>([]);
  const [loading, setLoading] = useState(true);

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

  return (
    <div className="mx-auto max-w-4xl px-4 py-8 sm:px-6">
      {/* Header */}
      <header className="mb-12 text-center">
        <h1 className="text-4xl font-bold tracking-tight">
          <a href="https://github.com/paradigmxyz/sfp" style={{ color: "var(--text)" }} className="hover:no-underline">
            sfp
          </a>
        </h1>
        <p className="mt-2 text-sm" style={{ color: "var(--text-dim)" }}>
          Does your LLM forget? Find out in 5 minutes.
        </p>
      </header>

      {/* Hero */}
      <section className="mb-8 text-center">
        <h2 className="mb-2 text-xl font-semibold">Forgetting Leaderboard</h2>
        <p className="text-sm" style={{ color: "var(--text-dim)" }}>
          Who can prevent the most catastrophic forgetting?{" "}
          <a href="https://github.com/paradigmxyz/sfp/blob/master/dev/LEADERBOARD.md" style={{ color: "var(--accent)" }}>
            Submit your method
          </a>{" "}
          and find out.
        </p>
      </section>

      {/* Forgetting Chart */}
      <section className="mb-8">
        <h2 className="mb-3 text-lg font-semibold">Forgetting Chart</h2>
        <ForgettingChart details={details} tasks={benchmark?.tasks ?? []} />
      </section>

      {/* Benchmark Info */}
      {benchmark && (
        <section className="mb-8">
          <h2 className="mb-3 text-lg font-semibold">Benchmark</h2>
          <BenchmarkInfo benchmark={benchmark} />
        </section>
      )}

      {/* Leaderboard */}
      <section className="mb-8">
        <h2 className="mb-3 text-lg font-semibold">Leaderboard</h2>
        <p className="mb-3 text-xs" style={{ color: "var(--text-dim)" }}>
          Score = 0.6 × retention + 0.4 × plasticity. Click column headers to sort.
        </p>
        <LeaderboardTable entries={entries} />
      </section>

      {/* Scatter Plot */}
      <section className="mb-8">
        <h2 className="mb-3 text-lg font-semibold">Retention vs Plasticity</h2>
        <ScatterPlot entries={entries} />
      </section>

      {/* Prometheus Charts */}
      <section className="mb-8">
        <h2 className="mb-3 text-lg font-semibold">Live Training</h2>
        <PrometheusCharts />
      </section>

      {/* How It Works */}
      <section className="mb-8">
        <h2 className="mb-3 text-lg font-semibold">How It Works</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {[
            {
              step: "1",
              title: "Write your method",
              desc: "Copy submissions/_example.py, write a loss function ending in _loss.",
            },
            {
              step: "2",
              title: "Submit",
              desc: "python submit.py submissions/my_method.py",
            },
            {
              step: "3",
              title: "Benchmark runs",
              desc: "A GPU spins up in the cloud. 3 seeds, ~45 min. No GPU needed on your end.",
            },
            {
              step: "4",
              title: "Score appears",
              desc: "Results land on this leaderboard automatically.",
            },
          ].map((item) => (
            <div
              key={item.step}
              className="rounded-md p-4"
              style={{ border: "1px solid var(--border)", background: "var(--surface)" }}
            >
              <div className="mb-1 text-xs font-bold" style={{ color: "var(--accent)" }}>
                Step {item.step}
              </div>
              <div className="mb-1 text-sm font-semibold">{item.title}</div>
              <div className="text-xs" style={{ color: "var(--text-dim)" }}>
                {item.desc}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Metrics Explainer */}
      <section className="mb-8">
        <h2 className="mb-3 text-lg font-semibold">Metrics</h2>
        <div className="space-y-3 text-sm" style={{ color: "var(--text-dim)" }}>
          <div>
            <span className="font-semibold" style={{ color: "var(--text)" }}>
              Retention
            </span>{" "}
            — Mean accuracy on previously-learned tasks after all training is complete. Measures how well
            your method prevents catastrophic forgetting.
          </div>
          <div>
            <span className="font-semibold" style={{ color: "var(--text)" }}>
              Plasticity
            </span>{" "}
            — Accuracy on the most recently trained task. Measures how well the model can still learn
            new things.
          </div>
          <div>
            <span className="font-semibold" style={{ color: "var(--text)" }}>
              Score
            </span>{" "}
            — 0.6 × retention + 0.4 × plasticity. Weighted composite that favors retention (since
            naive fine-tuning already achieves high plasticity).
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer
        className="border-t pt-6 pb-8 text-center text-xs"
        style={{ borderColor: "var(--border)", color: "var(--text-dim)" }}
      >
        <a href="https://github.com/paradigmxyz/sfp" style={{ color: "var(--accent)" }}>GitHub</a>
        {" · "}
        Built by <a href="https://paradigm.xyz" style={{ color: "var(--accent)" }}>Paradigm</a>
      </footer>
    </div>
  );
}
