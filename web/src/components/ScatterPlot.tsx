"use client";

import { useRef, useEffect } from "react";
import type { Entry } from "@/lib/types";
import { drawScatter } from "@/lib/charts";

export default function ScatterPlot({ entries }: { entries: Entry[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!canvasRef.current || entries.length === 0) return;
    drawScatter(canvasRef.current, entries);
  }, [entries]);

  if (entries.length === 0) {
    return (
      <div className="chart flex items-center justify-center" style={{ color: "var(--text-dim)" }}>
        No data yet.
      </div>
    );
  }

  return <canvas ref={canvasRef} className="chart chart-tall" />;
}
