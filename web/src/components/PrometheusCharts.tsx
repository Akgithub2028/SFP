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
        "sfp_training_loss",
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

  if (available === null) {
    return (
      <div className="text-sm" style={{ color: "var(--text-dim)" }}>
        Checking Prometheus...
      </div>
    );
  }

  if (!available) {
    return (
      <div
        className="rounded-md p-4 text-center text-sm"
        style={{ border: "1px solid var(--border)", color: "var(--text-dim)" }}
      >
        Prometheus not available — live training charts require a running Prometheus instance.
      </div>
    );
  }

  return (
    <div>
      {lossSeries.length > 0 ? (
        <canvas ref={canvasRef} className="chart chart-tall" />
      ) : (
        <div
          className="chart flex items-center justify-center text-sm"
          style={{ color: "var(--text-dim)" }}
        >
          No training loss data in the last 24h.
        </div>
      )}
    </div>
  );
}
