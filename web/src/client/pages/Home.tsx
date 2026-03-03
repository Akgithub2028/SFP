import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import type { Entry } from "@/lib/types";
import { fetchLeaderboard } from "@/lib/api";
import LeaderboardTable from "@/components/LeaderboardTable";
import ScatterPlot from "@/components/ScatterPlot";
import Nav from "@/components/Nav";
import Footer from "@/components/Footer";

export default function Home() {
  const [entries, setEntries] = useState<Entry[]>([]);

  useEffect(() => {
    fetchLeaderboard().then((lb) => {
      if (lb) setEntries(lb.entries);
    });
  }, []);

  return (
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <Nav />

      {/* ── Intro ── */}
      <section className="mb-4 text-sm leading-relaxed" style={{ color: "var(--text-dim)" }}>
        <p className="mb-2">
          Sequential fine-tuning of LLMs — alignment updates, domain adaptation, safety patches — causes
          capability regressions on previously learned tasks
          (<a href="https://arxiv.org/abs/2404.16789" style={{ color: "var(--accent)" }}>Wang et al., 2024</a>).
          This benchmark measures the retention–plasticity tradeoff under controlled conditions:
          same model, same data, same compute. The only variable is the training strategy.{" "}
          <Link to="/about" style={{ color: "var(--accent)" }}>Motivation →</Link>
        </p>
      </section>

      {/* ── Leaderboard + Scatter side by side ── */}
      <section className="mb-4 grid items-start gap-6 lg:grid-cols-2">
        <div>
          <h2 className="mb-1 text-lg font-semibold">Leaderboard</h2>
          <span className="text-xs" style={{ color: "var(--text-dim)" }}>
            Score = harmonic mean of retention &amp; plasticity
          </span>
          <LeaderboardTable entries={entries} />
        </div>
        <div>
          <ScatterPlot entries={entries} />
        </div>
      </section>

      {/* ── Submit CTA ── */}
      <section
        className="mb-4 flex items-center justify-between rounded-md p-5"
        style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
      >
        <div>
          <div className="text-sm font-semibold" style={{ color: "var(--text)" }}>
            Think you can do better?
          </div>
          <div className="text-xs" style={{ color: "var(--text-dim)" }}>
            Write a loss function, submit it, get scored on a cloud GPU. No GPU needed on your end.
          </div>
        </div>
        <Link
          to="/submit"
          className="rounded-md px-5 py-2 text-sm font-semibold"
          style={{ background: "var(--accent)", color: "#000" }}
        >
          Submit →
        </Link>
      </section>

      <Footer />
    </div>
  );
}
