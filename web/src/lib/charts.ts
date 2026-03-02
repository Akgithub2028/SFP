import type { DetailEntry, Entry, PromResult } from "./types";

export const COLORS = [
  "#22c55e",
  "#3b82f6",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#ec4899",
  "#14b8a6",
  "#f97316",
];

export const DASHES = [
  [],
  [6, 3],
  [2, 2],
  [8, 3, 2, 3],
  [4, 4],
];

export function initCanvas(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  const ctx = canvas.getContext("2d")!;
  ctx.scale(dpr, dpr);
  ctx.font = '11px "Berkeley Mono", "JetBrains Mono", monospace';
  return ctx;
}

interface Padding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

function defaultPad(): Padding {
  return { top: 30, right: 20, bottom: 40, left: 55 };
}

export function drawGrid(
  ctx: CanvasRenderingContext2D,
  pad: Padding,
  cw: number,
  ch: number,
  yLabels: number[]
) {
  const pw = cw - pad.left - pad.right;
  const ph = ch - pad.top - pad.bottom;
  const yMin = yLabels[0];
  const yMax = yLabels[yLabels.length - 1];

  ctx.strokeStyle = "#2a2a2a";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#888";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";

  for (const y of yLabels) {
    const py = pad.top + ph - ((y - yMin) / (yMax - yMin)) * ph;
    ctx.beginPath();
    ctx.moveTo(pad.left, py);
    ctx.lineTo(pad.left + pw, py);
    ctx.stroke();
    ctx.fillText(y.toFixed(2), pad.left - 8, py);
  }
}

export function drawForgettingChart(
  canvas: HTMLCanvasElement,
  details: DetailEntry[],
  tasks: string[]
) {
  const ctx = initCanvas(canvas);
  const rect = canvas.getBoundingClientRect();
  const cw = rect.width;
  const ch = rect.height;
  const pad = defaultPad();
  const pw = cw - pad.left - pad.right;
  const ph = ch - pad.top - pad.bottom;

  // Collect all accuracy values to determine y range
  const allVals: number[] = [];
  for (const entry of details) {
    if (!entry.per_seed?.length) continue;
    const seed = entry.per_seed[0];
    if (!seed.history?.length) continue;
    for (const stage of seed.history) {
      for (const task of tasks) {
        const metrics = stage[task];
        if (!metrics) continue;
        for (const v of Object.values(metrics)) {
          allVals.push(v);
        }
      }
    }
  }

  if (allVals.length === 0) return;

  const yMin = Math.max(0, Math.min(...allVals) - 0.05);
  const yMax = Math.min(1, Math.max(...allVals) + 0.05);

  // Y grid
  const yLabels: number[] = [];
  for (let y = Math.ceil(yMin * 10) / 10; y <= yMax + 0.001; y += 0.1) {
    yLabels.push(Math.round(y * 100) / 100);
  }
  if (yLabels.length < 2) yLabels.push(yMin, yMax);
  drawGrid(ctx, pad, cw, ch, yLabels);

  // X labels (tasks trained sequentially)
  const numStages = details[0]?.per_seed?.[0]?.history?.length ?? tasks.length;
  ctx.fillStyle = "#888";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i < numStages; i++) {
    const x = pad.left + (i / Math.max(1, numStages - 1)) * pw;
    const label = i < tasks.length ? tasks[i] : `step ${i}`;
    ctx.fillText(label, x, ch - pad.bottom + 8);
  }

  // Draw lines for each method
  for (let mi = 0; mi < details.length; mi++) {
    const entry = details[mi];
    if (!entry.per_seed?.length) continue;
    const seed = entry.per_seed[0];
    if (!seed.history?.length) continue;

    const color = COLORS[mi % COLORS.length];

    // For each task, draw its accuracy across stages
    for (let ti = 0; ti < tasks.length; ti++) {
      const task = tasks[ti];
      const dash = DASHES[ti % DASHES.length];

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.setLineDash(dash);
      ctx.beginPath();

      let started = false;
      for (let si = 0; si < seed.history.length; si++) {
        const metrics = seed.history[si]?.[task];
        if (!metrics) continue;
        const val = Object.values(metrics)[0] ?? 0;
        const x = pad.left + (si / Math.max(1, numStages - 1)) * pw;
        const y = pad.top + ph - ((val - yMin) / (yMax - yMin)) * ph;
        if (!started) {
          ctx.moveTo(x, y);
          started = true;
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Method label at right
    const lastStage = seed.history[seed.history.length - 1];
    const lastTask = tasks[tasks.length - 1];
    const lastMetrics = lastStage?.[lastTask];
    if (lastMetrics) {
      const lastVal = Object.values(lastMetrics)[0] ?? 0;
      const ly =
        pad.top + ph - ((lastVal - yMin) / (yMax - yMin)) * ph;
      ctx.fillStyle = color;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(entry.method, pad.left + pw + 4, ly);
    }
  }

  // Title
  ctx.fillStyle = "#888";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("Task accuracy over sequential training", cw / 2, 8);
}

export function drawScatter(
  canvas: HTMLCanvasElement,
  entries: Entry[]
) {
  const ctx = initCanvas(canvas);
  const rect = canvas.getBoundingClientRect();
  const cw = rect.width;
  const ch = rect.height;
  const pad = { top: 30, right: 80, bottom: 45, left: 55 };
  const pw = cw - pad.left - pad.right;
  const ph = ch - pad.top - pad.bottom;

  if (entries.length === 0) return;

  const retVals = entries.map((e) => e.retention_mean);
  const plasVals = entries.map((e) => e.plasticity_mean);
  const xMin = Math.max(0, Math.min(...plasVals) - 0.05);
  const xMax = Math.min(1, Math.max(...plasVals) + 0.05);
  const yMin = Math.max(0, Math.min(...retVals) - 0.05);
  const yMax = Math.min(1, Math.max(...retVals) + 0.05);

  // Grid
  const yLabels: number[] = [];
  for (let y = Math.ceil(yMin * 10) / 10; y <= yMax + 0.001; y += 0.1) {
    yLabels.push(Math.round(y * 100) / 100);
  }
  if (yLabels.length < 2) yLabels.push(yMin, yMax);
  drawGrid(ctx, pad, cw, ch, yLabels);

  // X axis labels
  ctx.fillStyle = "#888";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const xLabels: number[] = [];
  for (let x = Math.ceil(xMin * 10) / 10; x <= xMax + 0.001; x += 0.1) {
    xLabels.push(Math.round(x * 100) / 100);
  }
  for (const x of xLabels) {
    const px = pad.left + ((x - xMin) / (xMax - xMin)) * pw;
    ctx.fillText(x.toFixed(1), px, ch - pad.bottom + 8);
  }

  // Axis labels
  ctx.fillStyle = "#888";
  ctx.textAlign = "center";
  ctx.fillText("Plasticity", cw / 2, ch - 5);
  ctx.save();
  ctx.translate(12, ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Retention", 0, 0);
  ctx.restore();

  // Draw points
  for (let i = 0; i < entries.length; i++) {
    const e = entries[i];
    const px =
      pad.left + ((e.plasticity_mean - xMin) / (xMax - xMin)) * pw;
    const py =
      pad.top + ph - ((e.retention_mean - yMin) / (yMax - yMin)) * ph;
    const color = COLORS[i % COLORS.length];

    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(px, py, 5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(e.method, px + 8, py);
  }

  // Title
  ctx.fillStyle = "#888";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("Retention vs Plasticity", cw / 2, 8);
}

export function drawLossChart(
  canvas: HTMLCanvasElement,
  series: PromResult[]
) {
  const ctx = initCanvas(canvas);
  const rect = canvas.getBoundingClientRect();
  const cw = rect.width;
  const ch = rect.height;
  const pad = defaultPad();
  const pw = cw - pad.left - pad.right;
  const ph = ch - pad.top - pad.bottom;

  if (series.length === 0) return;

  // Collect all values
  let tMin = Infinity,
    tMax = -Infinity,
    vMin = Infinity,
    vMax = -Infinity;
  for (const s of series) {
    for (const [t, v] of s.values ?? []) {
      const val = parseFloat(v);
      if (isNaN(val)) continue;
      tMin = Math.min(tMin, t);
      tMax = Math.max(tMax, t);
      vMin = Math.min(vMin, val);
      vMax = Math.max(vMax, val);
    }
  }

  if (!isFinite(tMin)) return;

  // Add some padding to value range
  const vRange = vMax - vMin || 1;
  vMin = Math.max(0, vMin - vRange * 0.05);
  vMax = vMax + vRange * 0.05;

  // Y grid
  const yLabels: number[] = [];
  const yStep = (vMax - vMin) / 5;
  for (let i = 0; i <= 5; i++) {
    yLabels.push(vMin + yStep * i);
  }
  drawGrid(ctx, pad, cw, ch, yLabels);

  // X axis time labels
  ctx.fillStyle = "#888";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const tRange = tMax - tMin || 1;
  for (let i = 0; i <= 4; i++) {
    const t = tMin + (tRange / 4) * i;
    const px = pad.left + ((t - tMin) / tRange) * pw;
    const d = new Date(t * 1000);
    ctx.fillText(
      `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}`,
      px,
      ch - pad.bottom + 8
    );
  }

  // Draw series
  for (let si = 0; si < series.length; si++) {
    const s = series[si];
    const color = COLORS[si % COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    let started = false;
    for (const [t, v] of s.values ?? []) {
      const val = parseFloat(v);
      if (isNaN(val)) continue;
      const x = pad.left + ((t - tMin) / tRange) * pw;
      const y = pad.top + ph - ((val - vMin) / (vMax - vMin)) * ph;
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Label
    const label = s.metric?.method ?? s.metric?.job ?? `series ${si}`;
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    const lastPoint = s.values?.[s.values.length - 1];
    if (lastPoint) {
      const ly =
        pad.top +
        ph -
        ((parseFloat(lastPoint[1]) - vMin) / (vMax - vMin)) * ph;
      ctx.fillText(label, pad.left + pw + 4, ly);
    }
  }

  // Title
  ctx.fillStyle = "#888";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("Training Loss (24h)", cw / 2, 8);
}
