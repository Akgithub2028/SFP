import type { DetailEntry, Entry } from "./types";

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
  let yMax = yLabels[yLabels.length - 1];
  if (yMax <= yMin) yMax = yMin + 0.1;

  const styles = getComputedStyle(document.documentElement);
  const borderColor = styles.getPropertyValue('--border').trim() || '#2a2a2a';
  const dimColor = styles.getPropertyValue('--text-dim').trim() || '#888';

  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 1;
  ctx.fillStyle = dimColor;
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

// Methods to highlight in the forgetting chart (narrative: problem → baseline → solution)
const HIGHLIGHT_METHODS = ["naive", "replay", "sfp"];
const HIGHLIGHT_COLORS: Record<string, string> = {
  naive: "#ef4444",   // red — the problem
  replay: "#3b82f6",  // blue — simple baseline
  sfp: "#22c55e",     // green — our solution
};

export function drawForgettingChart(
  canvas: HTMLCanvasElement,
  details: DetailEntry[],
  tasks: string[]
) {
  const ctx = initCanvas(canvas);
  const rect = canvas.getBoundingClientRect();
  const cw = rect.width;
  const ch = rect.height;
  const pad = { top: 30, right: 80, bottom: 45, left: 55 };
  const pw = cw - pad.left - pad.right;
  const ph = ch - pad.top - pad.bottom;

  const styles = getComputedStyle(document.documentElement);
  const dimColor = styles.getPropertyValue('--text-dim').trim() || '#888';

  // Filter to highlight methods only
  const shown = details.filter(d => HIGHLIGHT_METHODS.includes(d.method));
  if (shown.length === 0) return;

  const numStages = shown[0]?.per_seed?.[0]?.history?.length ?? tasks.length;

  // Compute mean retention at each stage for each method:
  // retention(stage) = mean accuracy across all tasks seen up to that stage
  type MethodCurve = { method: string; points: number[] };
  const curves: MethodCurve[] = [];

  for (const entry of shown) {
    if (!entry.per_seed?.length) continue;
    const points: number[] = [];

    for (let si = 0; si < numStages; si++) {
      let totalAcc = 0, totalCount = 0;
      for (const seed of entry.per_seed) {
        const stage = seed.history?.[si];
        if (!stage) continue;
        for (const task of tasks.slice(0, si + 1)) {
          const metrics = stage[task];
          if (!metrics) continue;
          totalAcc += Object.values(metrics)[0] ?? 0;
          totalCount++;
        }
      }
      points.push(totalCount > 0 ? totalAcc / totalCount : 0);
    }
    curves.push({ method: entry.method, points });
  }

  // Y range from all points
  const allVals = curves.flatMap(c => c.points);
  if (allVals.length === 0) return;

  let yMin = Math.max(0, Math.min(...allVals) - 0.05);
  let yMax = Math.min(1, Math.max(...allVals) + 0.05);
  const ySpan = yMax - yMin;
  if (ySpan < 0.2) {
    const mid = (yMax + yMin) / 2;
    yMin = Math.max(0, mid - 0.1);
    yMax = Math.min(1, mid + 0.1);
  }

  // Y grid
  const yLabels: number[] = [];
  for (let y = Math.ceil(yMin * 10) / 10; y <= yMax + 0.001; y += 0.1) {
    yLabels.push(Math.round(y * 100) / 100);
  }
  if (yLabels.length < 2) yLabels.push(yMin, yMax);
  drawGrid(ctx, pad, cw, ch, yLabels);

  // X labels — "After task N" with task name
  ctx.fillStyle = dimColor;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let i = 0; i < numStages; i++) {
    const x = pad.left + (i / Math.max(1, numStages - 1)) * pw;
    const label = i < tasks.length ? `+${tasks[i]}` : `+task ${i}`;
    ctx.fillText(label, x, ch - pad.bottom + 8);
  }

  // Axis label
  ctx.fillText("task trained", cw / 2, ch - 5);

  // Y axis label
  ctx.save();
  ctx.translate(12, pad.top + ph / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textAlign = "center";
  ctx.fillText("mean accuracy (all tasks)", 0, 0);
  ctx.restore();

  // Draw curves — sorted so sfp draws on top
  const drawOrder = ["naive", "replay", "sfp"];
  const sorted = [...curves].sort(
    (a, b) => drawOrder.indexOf(a.method) - drawOrder.indexOf(b.method)
  );

  for (const curve of sorted) {
    const color = HIGHLIGHT_COLORS[curve.method] ?? COLORS[0];
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.5;
    ctx.setLineDash([]);
    ctx.beginPath();

    let started = false;
    for (let si = 0; si < curve.points.length; si++) {
      const x = pad.left + (si / Math.max(1, numStages - 1)) * pw;
      const y = pad.top + ph - ((curve.points[si] - yMin) / (yMax - yMin)) * ph;
      if (!started) { ctx.moveTo(x, y); started = true; }
      else { ctx.lineTo(x, y); }
    }
    ctx.stroke();

    // Dots at each point
    ctx.fillStyle = color;
    for (let si = 0; si < curve.points.length; si++) {
      const x = pad.left + (si / Math.max(1, numStages - 1)) * pw;
      const y = pad.top + ph - ((curve.points[si] - yMin) / (yMax - yMin)) * ph;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }

    // Label at right
    const lastVal = curve.points[curve.points.length - 1];
    const ly = pad.top + ph - ((lastVal - yMin) / (yMax - yMin)) * ph;
    ctx.fillStyle = color;
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.font = 'bold 11px "Berkeley Mono", "JetBrains Mono", monospace';
    ctx.fillText(curve.method, pad.left + pw + 6, ly);
    ctx.font = '11px "Berkeley Mono", "JetBrains Mono", monospace';
  }

  // Title
  ctx.fillStyle = dimColor;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("Mean accuracy as tasks are added", cw / 2, 8);
}

export function drawScatter(
  canvas: HTMLCanvasElement,
  entries: Entry[]
) {
  const ctx = initCanvas(canvas);
  const rect = canvas.getBoundingClientRect();
  const cw = rect.width;
  const ch = rect.height;
  const pad = { top: 20, right: 110, bottom: 55, left: 70 };
  const pw = cw - pad.left - pad.right;
  const ph = ch - pad.top - pad.bottom;

  const styles = getComputedStyle(document.documentElement);
  const dimColor = styles.getPropertyValue('--text-dim').trim() || '#888';

  if (entries.length === 0) return;

  const retVals = entries.map((e) => e.retention_mean);
  const plasVals = entries.map((e) => e.plasticity_mean);
  let xMin = Math.max(0, Math.min(...plasVals) - 0.05);
  let xMax = Math.min(1, Math.max(...plasVals) + 0.05);
  let yMin = Math.max(0, Math.min(...retVals) - 0.05);
  let yMax = Math.min(1, Math.max(...retVals) + 0.05);

  // Ensure minimum span of 0.2
  const xSpan = xMax - xMin;
  if (xSpan < 0.2) {
    const mid = (xMax + xMin) / 2;
    xMin = Math.max(0, mid - 0.1);
    xMax = Math.min(1, mid + 0.1);
  }
  const ySpan = yMax - yMin;
  if (ySpan < 0.2) {
    const mid = (yMax + yMin) / 2;
    yMin = Math.max(0, mid - 0.1);
    yMax = Math.min(1, mid + 0.1);
  }

  // Grid
  const yLabels: number[] = [];
  for (let y = Math.ceil(yMin * 10) / 10; y <= yMax + 0.001; y += 0.1) {
    yLabels.push(Math.round(y * 100) / 100);
  }
  if (yLabels.length < 2) yLabels.push(yMin, yMax);
  drawGrid(ctx, pad, cw, ch, yLabels);

  // X axis labels
  ctx.fillStyle = dimColor;
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
  ctx.fillStyle = dimColor;
  ctx.textAlign = "center";
  ctx.fillText("Plasticity", cw / 2, ch - pad.bottom + 24);
  ctx.save();
  ctx.translate(14, ch / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("Retention", 0, 0);
  ctx.restore();

  // Draw points
  const labels: { x: number; y: number; method: string; color: string }[] = [];
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

    labels.push({ x: px, y: py, method: e.method, color });
  }

  // Draw labels with collision avoidance
  labels.sort((a, b) => a.y - b.y);
  const placed: { x: number; y: number; w: number; align: "left" | "right" }[] = [];
  ctx.textBaseline = "middle";

  for (const lbl of labels) {
    const textWidth = ctx.measureText(lbl.method).width;
    const rightEdge = cw - 4;
    const fitsRight = lbl.x + 10 + textWidth <= rightEdge;
    const align = fitsRight ? "left" as const : "right" as const;
    const lx = fitsRight ? lbl.x + 10 : lbl.x - 10;

    // Find a y that doesn't collide with any placed label
    let ly = lbl.y;
    let collides = true;
    while (collides) {
      collides = false;
      for (const p of placed) {
        if (Math.abs(ly - p.y) < 13) {
          ly = p.y + 13;
          collides = true;
        }
      }
    }
    placed.push({ x: lx, y: ly, w: textWidth, align });

    ctx.fillStyle = lbl.color;
    ctx.textAlign = align;
    ctx.fillText(lbl.method, lx, ly);
  }
}
