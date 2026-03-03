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
        <ul className="list-inside list-disc space-y-0.5">
          <li>Fine-tune an LLM on tasks A, B, C in sequence — <strong style={{ color: "var(--text)" }}>how much of A does it forget?</strong></li>
          <li>Every method below is a single loss function. Same model, same data, same compute.</li>
          <li>The only variable is your training strategy. <Link to="/about" style={{ color: "var(--accent)" }}>Read more →</Link></li>
        </ul>
      </section>

      {/* ── Leaderboard + Scatter side by side ── */}
      <section className="mb-4 grid items-start gap-6 lg:grid-cols-2">
        <div>
          <span className="text-xs" style={{ color: "var(--text-dim)" }}>
            Score = 0.6 × retention + 0.4 × plasticity
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
