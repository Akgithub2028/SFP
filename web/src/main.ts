/**
 * sfp leaderboard — fetch data from GitHub, render table.
 */

const DATA_URL =
  "https://raw.githubusercontent.com/paradigmxyz/sfp/master/leaderboard.json";

interface Entry {
  rank?: number;
  method: string;
  description: string;
  score_mean: number;
  score_std: number;
  retention_mean: number;
  retention_std: number;
  plasticity_mean: number;
  plasticity_std: number;
  contributor: string;
  date: string;
  pr?: string;
  commit?: string;
}

interface LeaderboardData {
  updated: string;
  benchmark: {
    model: string;
    tasks: string[];
    steps_per_task: number;
    lora_rank: number;
    seeds: number[];
    memory: number;
  };
  entries: Entry[];
}

function fmt(val: number, std?: number): string {
  const s = val.toFixed(4);
  if (std !== undefined && std > 0) {
    return `${s} <span class="std">±${std.toFixed(4)}</span>`;
  }
  return s;
}

function medalIcon(rank: number): string {
  if (rank === 1) return '<span class="medal-1">🥇</span>';
  if (rank === 2) return '<span class="medal-2">🥈</span>';
  if (rank === 3) return '<span class="medal-3">🥉</span>';
  return `${rank}`;
}

function renderTable(entries: Entry[]): void {
  const tbody = document.getElementById("leaderboard-body")!;
  const empty = document.getElementById("empty-state")!;

  if (entries.length === 0) {
    empty.style.display = "block";
    return;
  }

  empty.style.display = "none";

  // Sort by score descending
  const sorted = [...entries].sort(
    (a, b) => b.score_mean - a.score_mean
  );

  tbody.innerHTML = sorted
    .map((e, i) => {
      const rank = i + 1;
      const prLink = e.pr
        ? `<a href="${e.pr}">#${e.pr.split("/").pop()}</a>`
        : "—";
      return `<tr>
        <td class="rank">${medalIcon(rank)}</td>
        <td class="score num">${fmt(e.score_mean, e.score_std)}</td>
        <td class="num">${fmt(e.retention_mean, e.retention_std)}</td>
        <td class="num">${fmt(e.plasticity_mean, e.plasticity_std)}</td>
        <td class="method">${esc(e.method)}</td>
        <td class="desc" title="${esc(e.description)}">${esc(e.description)}</td>
        <td>${esc(e.contributor)}</td>
        <td>${esc(e.date)}</td>
        <td>${prLink}</td>
      </tr>`;
    })
    .join("");
}

function renderInfo(data: LeaderboardData): void {
  const b = data.benchmark;
  setText("info-model", b.model.split("/").pop() || b.model);
  setText("info-tasks", b.tasks.join(" → "));
  setText("info-steps", `${b.steps_per_task}/task`);
  setText("info-memory", `M=${b.memory}`);
  setText("info-seeds", b.seeds.join(", "));
  setText("info-updated", data.updated);
}

function setText(id: string, text: string): void {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function esc(s: string): string {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function setupSort(entries: Entry[]): void {
  document.querySelectorAll("th.sortable").forEach((th) => {
    th.addEventListener("click", () => {
      const key = (th as HTMLElement).dataset.sort as keyof Entry;
      const meanKey = `${key}_mean` as keyof Entry;
      const sorted = [...entries].sort((a, b) => {
        const av = (a[meanKey] as number) ?? 0;
        const bv = (b[meanKey] as number) ?? 0;
        return bv - av;
      });
      renderTable(sorted);
    });
  });
}

async function init(): Promise<void> {
  try {
    const res = await fetch(DATA_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data: LeaderboardData = await res.json();
    renderInfo(data);
    renderTable(data.entries);
    setupSort(data.entries);
  } catch (err) {
    console.error("Failed to load leaderboard data:", err);
    // Show empty state
    const empty = document.getElementById("empty-state")!;
    empty.style.display = "block";
    empty.innerHTML =
      "<p>Loading leaderboard data...</p><p style='color: var(--text-dim); font-size: 0.8rem;'>If this persists, the data may not be available yet.</p>";
  }
}

init();
