"use client";

import { useRef, useEffect, useState } from "react";
import { promInstant, promRange } from "@/lib/api";
import { drawLossChart } from "@/lib/charts";
import type { PromResult } from "@/lib/types";

export default function PrometheusCharts() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [available, setAvailable] = useState<boolean | null>(null);
  const [lossSeries, setLossSeries] = useState<PromResult[]>([]);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      const up = await promInstant("up");
      if (cancelled) return;
      if (up.length === 0) {
        setAvailable(false);
        return;
      }
      setAvailable(true);

      const now = Math.floor(Date.now() / 1000);
      const start = (now - 86400).toString();
      const end = now.toString();
      const series = await promRange(
        "sfp_train_loss",
        start,
        end,
        "60"
      );
      if (!cancelled) setLossSeries(series);
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!canvasRef.current || lossSeries.length === 0) return;
    drawLossChart(canvasRef.current, lossSeries);
  }, [lossSeries]);

  if (available === null || !available || lossSeries.length === 0) {
    return null;
  }

  return (
    <section className="mb-8">
      <h2 className="mb-3 text-lg font-semibold">Live Training</h2>
      <canvas ref={canvasRef} className="chart chart-tall" />
    </section>
  );
}
