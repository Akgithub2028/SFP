import type { Entry } from "@/lib/types";

const MEDALS = ["🥇", "🥈", "🥉"];

export default function LeaderboardTable({ entries }: { entries: Entry[] }) {
  const sorted = [...entries].sort((a, b) => b.score_mean - a.score_mean);

  if (entries.length === 0) {
    return (
      <div
        className="rounded-md p-8 text-center"
        style={{ border: "1px solid var(--border)", color: "var(--text-dim)" }}
      >
        <p>No submissions yet.</p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Method</th>
            <th>Contributor</th>
            <th>Score</th>
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
              <td
                style={{
                  fontVariantNumeric: "tabular-nums",
                  color: i === 0 ? "var(--accent)" : undefined,
                }}
              >
                {e.score_mean.toFixed(4)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
