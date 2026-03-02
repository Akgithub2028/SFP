import { useState } from "react";
import type { Entry } from "@/lib/types";

type SortKey = "score_mean" | "retention_mean" | "plasticity_mean";

const MEDALS = ["🥇", "🥈", "🥉"];

function fmt(v: number, std?: number): string {
  const s = v.toFixed(4);
  if (std != null && std > 0 && std.toFixed(4) !== "0.0000") {
    return `${s} ±${std.toFixed(4)}`;
  }
  return s;
}

export default function LeaderboardTable({ entries }: { entries: Entry[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("score_mean");

  const sorted = [...entries].sort((a, b) => b[sortKey] - a[sortKey]);

  if (entries.length === 0) {
    return (
      <div
        className="rounded-md p-8 text-center"
        style={{ border: "1px solid var(--border)", color: "var(--text-dim)" }}
      >
        <p className="mb-2">No submissions yet.</p>
        <p className="text-sm">
          Add your method to <code className="px-1" style={{ background: "var(--surface)" }}>methods.py</code> and run{" "}
          <code className="px-1" style={{ background: "var(--surface)" }}>python submit.py</code>
        </p>
      </div>
    );
  }

  const headerBtn = (key: SortKey, label: string) => (
    <th
      className="cursor-pointer select-none"
      onClick={() => setSortKey(key)}
      style={{ color: sortKey === key ? "var(--accent)" : undefined }}
    >
      {label} {sortKey === key ? "↓" : ""}
    </th>
  );

  return (
    <div className="overflow-x-auto">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Method</th>
            <th>Contributor</th>
            {headerBtn("score_mean", "Score")}
            {headerBtn("retention_mean", "Retention")}
            {headerBtn("plasticity_mean", "Plasticity")}
            <th>Date</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((e, i) => (
            <tr key={e.method}>
              <td style={{ fontVariantNumeric: "tabular-nums" }}>
                {i < 3 ? MEDALS[i] : i + 1}
              </td>
              <td className="font-semibold" style={{ color: "var(--text)" }}>
                {e.pr ? (
                  <a
                    href={e.pr}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: "var(--accent)", textDecoration: "underline" }}
                  >
                    {e.method}
                  </a>
                ) : (
                  e.method
                )}
              </td>
              <td style={{ color: "var(--text-dim)" }}>{e.contributor}</td>
              <td style={{ fontVariantNumeric: "tabular-nums", color: i === 0 && sortKey === "score_mean" ? "var(--accent)" : undefined }}>
                {fmt(e.score_mean, e.score_std)}
              </td>
              <td style={{ fontVariantNumeric: "tabular-nums" }}>
                {fmt(e.retention_mean, e.retention_std)}
              </td>
              <td style={{ fontVariantNumeric: "tabular-nums" }}>
                {fmt(e.plasticity_mean, e.plasticity_std)}
              </td>
              <td style={{ color: "var(--text-dim)", fontSize: "0.8rem" }}>
                {e.date ? new Date(e.date).toLocaleDateString() : "—"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
