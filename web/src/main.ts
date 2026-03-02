/**
 * sfp leaderboard — fetch leaderboard from GitHub, loss curves from Prometheus.
 */

const DATA_URL =
  "https://raw.githubusercontent.com/paradigmxyz/sfp/master/leaderboard.json";
const API_BASE = location.port === "5173" ? "http://localhost:8090" : "";

// Prometheus is proxied through nginx at /api/v1/
const PROM_API = "/api/v1";

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

// ---------------------------------------------------------------------------
// Leaderboard table
// ---------------------------------------------------------------------------

function fmt(val: number, std?: number): string {
  const s = val.toFixed(4);
  if (std !== undefined && std > 0) {
    return `${s} <span class="std">+/-${std.toFixed(4)}</span>`;
  }
  return s;
}

function medalIcon(rank: number): string {
  if (rank === 1) return '<span class="medal-1">1</span>';
  if (rank === 2) return '<span class="medal-2">2</span>';
  if (rank === 3) return '<span class="medal-3">3</span>';
  return `${rank}`;
}

function esc(s: string): string {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function renderTable(entries: Entry[]): void {
  const tbody = document.getElementById("leaderboard-body")!;
  const empty = document.getElementById("empty-state")!;

  if (entries.length === 0) {
    empty.style.display = "block";
    return;
  }
  empty.style.display = "none";

  const sorted = [...entries].sort((a, b) => b.score_mean - a.score_mean);
  tbody.innerHTML = sorted
    .map((e, i) => {
      const rank = i + 1;
      const prLink = e.pr
        ? `<a href="${e.pr}">#${e.pr.split("/").pop()}</a>`
        : "";
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
  setText("info-tasks", b.tasks.join(" -> "));
  setText("info-steps", `${b.steps_per_task}/task`);
  setText("info-memory", `M=${b.memory}`);
  setText("info-seeds", b.seeds.join(", "));
  setText("info-updated", data.updated);
}

function setText(id: string, text: string): void {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
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

// ---------------------------------------------------------------------------
// Prometheus charts (loss curves, eval scores)
// ---------------------------------------------------------------------------

interface PromResult {
  metric: Record<string, string>;
  values: [number, string][];
}

async function promQuery(query: string, start: string, end: string, step: string): Promise<PromResult[]> {
  const params = new URLSearchParams({ query, start, end, step });
  try {
    const res = await fetch(`${PROM_API}/query_range?${params}`);
    if (!res.ok) return [];
    const json = await res.json();
    return json.data?.result ?? [];
  } catch {
    return [];
  }
}

async function promInstant(query: string): Promise<PromResult[]> {
  try {
    const res = await fetch(`${PROM_API}/query?query=${encodeURIComponent(query)}`);
    if (!res.ok) return [];
    const json = await res.json();
    return json.data?.result ?? [];
  } catch {
    return [];
  }
}

function drawChart(canvas: HTMLCanvasElement, series: { label: string; data: [number, number][] }[]): void {
  const ctx = canvas.getContext("2d");
  if (!ctx || series.length === 0) return;

  const w = canvas.width = canvas.offsetWidth * 2;
  const h = canvas.height = canvas.offsetHeight * 2;
  ctx.scale(2, 2);
  const cw = w / 2, ch = h / 2;
  const pad = { top: 20, right: 20, bottom: 30, left: 50 };
  const pw = cw - pad.left - pad.right;
  const ph = ch - pad.top - pad.bottom;

  // Find ranges
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const s of series) {
    for (const [x, y] of s.data) {
      xMin = Math.min(xMin, x); xMax = Math.max(xMax, x);
      yMin = Math.min(yMin, y); yMax = Math.max(yMax, y);
    }
  }
  if (xMin === xMax) xMax = xMin + 1;
  if (yMin === yMax) { yMin -= 0.1; yMax += 0.1; }
  yMin = Math.max(0, yMin - (yMax - yMin) * 0.05);
  yMax = yMax + (yMax - yMin) * 0.05;

  const toX = (v: number) => pad.left + ((v - xMin) / (xMax - xMin)) * pw;
  const toY = (v: number) => pad.top + (1 - (v - yMin) / (yMax - yMin)) * ph;

  // Background
  ctx.fillStyle = "#0a0a0a";
  ctx.fillRect(0, 0, cw, ch);

  // Grid
  ctx.strokeStyle = "#1a1a1a";
  ctx.lineWidth = 0.5;
  for (let i = 0; i <= 4; i++) {
    const y = pad.top + (ph / 4) * i;
    ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(cw - pad.right, y); ctx.stroke();
  }

  // Axes labels
  ctx.fillStyle = "#888";
  ctx.font = "10px monospace";
  ctx.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const v = yMax - ((yMax - yMin) / 4) * i;
    ctx.fillText(v.toFixed(2), pad.left - 5, pad.top + (ph / 4) * i + 4);
  }
  ctx.textAlign = "center";
  ctx.fillText(`${Math.round(xMin)}`, toX(xMin), ch - 5);
  ctx.fillText(`${Math.round(xMax)}`, toX(xMax), ch - 5);

  // Series
  const colors = ["#22c55e", "#60a5fa", "#f59e0b", "#ef4444", "#a78bfa", "#ec4899"];
  series.forEach((s, idx) => {
    ctx.strokeStyle = colors[idx % colors.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    s.data.forEach(([x, y], i) => {
      const px = toX(x), py = toY(y);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // Legend
    const lx = pad.left + 10 + idx * 80;
    ctx.fillStyle = colors[idx % colors.length];
    ctx.fillRect(lx, 5, 12, 12);
    ctx.fillStyle = "#ccc";
    ctx.textAlign = "left";
    ctx.font = "9px monospace";
    ctx.fillText(s.label, lx + 16, 14);
  });
}

async function loadCharts(): Promise<void> {
  const chartsEl = document.getElementById("charts")!;

  // Check if Prometheus is reachable
  const test = await promInstant("up");
  if (test.length === 0) {
    chartsEl.innerHTML = '<p class="chart-note">Prometheus not available (charts require dev-georgios).</p>';
    return;
  }

  // Loss curves (last 24h)
  const now = new Date();
  const dayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000);
  const lossSeries = await promQuery(
    "sfp_train_loss",
    dayAgo.toISOString(),
    now.toISOString(),
    "30s",
  );

  if (lossSeries.length > 0) {
    const lossCanvas = document.createElement("canvas");
    lossCanvas.className = "chart";
    const lossTitle = document.createElement("h3");
    lossTitle.textContent = "Training Loss (last 24h)";
    chartsEl.appendChild(lossTitle);
    chartsEl.appendChild(lossCanvas);

    const chartData = lossSeries.map((r) => ({
      label: `${r.metric.method || "?"} / ${r.metric.task || "?"}`,
      data: r.values.map(([t, v]) => [t, parseFloat(v)] as [number, number]),
    }));
    drawChart(lossCanvas, chartData);
  }

  // Eval scores
  const evalSeries = await promInstant("sfp_eval_score");
  if (evalSeries.length > 0) {
    const evalDiv = document.createElement("div");
    evalDiv.className = "eval-metrics";
    const evalTitle = document.createElement("h3");
    evalTitle.textContent = "Latest Eval Scores";
    chartsEl.appendChild(evalTitle);
    chartsEl.appendChild(evalDiv);

    evalDiv.innerHTML = `<table class="eval-table">
      <thead><tr><th>Method</th><th>Task</th><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        ${evalSeries
          .map((r) => `<tr>
            <td>${esc(r.metric.method || "?")}</td>
            <td>${esc(r.metric.task || "?")}</td>
            <td>${esc(r.metric.metric || "?")}</td>
            <td class="num">${parseFloat(r.values?.[0]?.[1] ?? (r as any).value?.[1] ?? "0").toFixed(4)}</td>
          </tr>`)
          .join("")}
      </tbody>
    </table>`;
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function fetchLeaderboard(): Promise<Entry[]> {
  // Try API first (has latest data from SQLite)
  try {
    const res = await fetch(`${API_BASE}/leaderboard`);
    if (res.ok) {
      const data = await res.json();
      return data.entries ?? [];
    }
  } catch { /* fall through */ }

  // Fall back to static JSON from GitHub
  try {
    const res = await fetch(DATA_URL);
    if (res.ok) {
      const data = await res.json();
      return data.entries ?? [];
    }
  } catch { /* fall through */ }

  return [];
}

async function init(): Promise<void> {
  // Load leaderboard info (static)
  try {
    const res = await fetch(DATA_URL);
    if (res.ok) {
      const data: LeaderboardData = await res.json();
      renderInfo(data);
    }
  } catch { /* ignore */ }

  // Load entries (API -> GitHub fallback)
  const entries = await fetchLeaderboard();
  renderTable(entries);
  setupSort(entries);

  // Load Prometheus charts
  await loadCharts();
}

init();
