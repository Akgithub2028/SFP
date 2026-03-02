import type { DetailEntry, Entry, LiveSeries, PromResult } from "./types";

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
  return { top: 30, right: 80, bottom: 40, left: 55 };
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
  const pad = { top: 30, right: 80, bottom: 45, left: 55 };
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
  ctx.fillStyle = dimColor;
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

  const styles = getComputedStyle(document.documentElement);
  const dimColor = styles.getPropertyValue('--text-dim').trim() || '#888';

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
  ctx.fillStyle = dimColor;
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
  ctx.fillStyle = dimColor;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("Training Loss (24h)", cw / 2, 8);
}

export function joinLiveSeries(
  lossSeries: PromResult[],
  stepSeries: PromResult[]
): LiveSeries[] {
  // Build a lookup: method+seed+timestamp → step
  const stepMap = new Map<string, Map<number, number>>();
  for (const s of stepSeries) {
    const key = `${s.metric.method}|${s.metric.seed}`;
    if (!stepMap.has(key)) stepMap.set(key, new Map());
    const m = stepMap.get(key)!;
    for (const [t, v] of s.values ?? []) {
      m.set(t, parseFloat(v));
    }
  }

  // For each loss series, look up the step at each timestamp
  // Group by method, average across seeds at each step
  const methodPoints = new Map<string, Map<number, { sum: number; n: number }>>();

  for (const s of lossSeries) {
    const method = s.metric.method;
    const key = `${method}|${s.metric.seed}`;
    const steps = stepMap.get(key);
    if (!steps) continue;

    if (!methodPoints.has(method)) methodPoints.set(method, new Map());
    const mp = methodPoints.get(method)!;

    for (const [t, v] of s.values ?? []) {
      const loss = parseFloat(v);
      const step = steps.get(t);
      if (step == null || isNaN(loss)) continue;
      const rounded = Math.round(step / 10) * 10; // bucket by 10 steps
      const existing = mp.get(rounded);
      if (existing) {
        existing.sum += loss;
        existing.n++;
      } else {
        mp.set(rounded, { sum: loss, n: 1 });
      }
    }
  }

  const result: LiveSeries[] = [];
  for (const [method, points] of methodPoints) {
    const sorted = [...points.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([step, { sum, n }]) => ({ step, loss: sum / n }));
    if (sorted.length > 0) {
      result.push({ method, points: sorted });
    }
  }
  return result.sort((a, b) => a.method.localeCompare(b.method));
}

export function drawLiveTraining(
  canvas: HTMLCanvasElement,
  series: LiveSeries[],
  stepsPerTask: number = 1000,
  tasks: string[] = ["math", "code", "ifeval"]
) {
  const ctx = initCanvas(canvas);
  const rect = canvas.getBoundingClientRect();
  const cw = rect.width;
  const ch = rect.height;
  const pad = { top: 30, right: 100, bottom: 45, left: 55 };
  const pw = cw - pad.left - pad.right;
  const ph = ch - pad.top - pad.bottom;

  const styles = getComputedStyle(document.documentElement);
  const dimColor = styles.getPropertyValue("--text-dim").trim() || "#888";
  const borderColor = styles.getPropertyValue("--border").trim() || "#2a2a2a";

  if (series.length === 0) return;

  // Determine ranges
  const totalSteps = stepsPerTask * tasks.length;
  const xMin = 0;
  const xMax = Math.max(
    totalSteps,
    ...series.map((s) => s.points[s.points.length - 1]?.step ?? 0)
  );

  let vMin = Infinity;
  let vMax = -Infinity;
  for (const s of series) {
    for (const p of s.points) {
      vMin = Math.min(vMin, p.loss);
      vMax = Math.max(vMax, p.loss);
    }
  }
  if (!isFinite(vMin)) return;
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

  // Task boundary vertical lines
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 1;
  for (let ti = 1; ti < tasks.length; ti++) {
    const bx = pad.left + ((stepsPerTask * ti) / xMax) * pw;
    ctx.beginPath();
    ctx.moveTo(bx, pad.top);
    ctx.lineTo(bx, pad.top + ph);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // Task labels at top
  ctx.fillStyle = dimColor;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.font = '10px "Berkeley Mono", "JetBrains Mono", monospace';
  for (let ti = 0; ti < tasks.length; ti++) {
    const mid = stepsPerTask * ti + stepsPerTask / 2;
    const tx = pad.left + (mid / xMax) * pw;
    ctx.fillText(tasks[ti], tx, pad.top + 4);
  }
  ctx.font = '11px "Berkeley Mono", "JetBrains Mono", monospace';

  // X axis step labels
  ctx.fillStyle = dimColor;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const xStepSize = stepsPerTask;
  for (let x = 0; x <= xMax; x += xStepSize) {
    const px = pad.left + (x / xMax) * pw;
    ctx.fillText(x.toString(), px, ch - pad.bottom + 8);
  }

  // Axis label
  ctx.fillText("Step", cw / 2, ch - 5);

  // Draw series
  for (let si = 0; si < series.length; si++) {
    const s = series[si];
    const color = COLORS[si % COLORS.length];
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    let started = false;
    for (const p of s.points) {
      const x = pad.left + (p.step / xMax) * pw;
      const y = pad.top + ph - ((p.loss - vMin) / (vMax - vMin)) * ph;
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    // Label at last point
    const last = s.points[s.points.length - 1];
    if (last) {
      const ly = pad.top + ph - ((last.loss - vMin) / (vMax - vMin)) * ph;
      ctx.fillStyle = color;
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(s.method, pad.left + pw + 4, ly);
    }
  }

  // Title
  ctx.fillStyle = dimColor;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("Training Loss vs Step", cw / 2, 8);
}
