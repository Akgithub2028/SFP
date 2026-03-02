"use client";

import { useRef, useEffect, useState, useCallback } from "react";
import { promRange } from "@/lib/api";
import { joinLiveSeries, drawLiveTraining } from "@/lib/charts";
import type { LiveSeries } from "@/lib/types";

const POLL_INTERVAL = 30_000;

export default function PrometheusCharts() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [series, setSeries] = useState<LiveSeries[]>([]);

  const fetchData = useCallback(async () => {
    const now = Math.floor(Date.now() / 1000);
    const start = (now - 86400).toString();
    const end = now.toString();

    const [lossSeries, stepSeries] = await Promise.all([
      promRange("sfp_train_loss", start, end, "30"),
      promRange("sfp_train_step", start, end, "30"),
    ]);

    if (lossSeries.length === 0 || stepSeries.length === 0) return;
    setSeries(joinLiveSeries(lossSeries, stepSeries));
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, POLL_INTERVAL);
    return () => clearInterval(id);
  }, [fetchData]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || series.length === 0) return;

    const draw = () => drawLiveTraining(canvas, series);
    draw();

    const obs = new ResizeObserver(draw);
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [series]);

  if (series.length === 0) return null;

  return (
    <section className="mb-8">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-lg font-semibold">Live Training</h2>
        <span className="text-xs" style={{ color: "var(--text-dim)" }}>
          Updates every 30s · {series.length} method{series.length !== 1 ? "s" : ""}
        </span>
      </div>
      <canvas ref={canvasRef} className="chart chart-tall" />
    </section>
  );
}
