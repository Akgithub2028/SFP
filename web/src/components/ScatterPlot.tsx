"use client";

import { useRef, useEffect } from "react";
import type { Entry } from "@/lib/types";
import { drawScatter } from "@/lib/charts";

export default function ScatterPlot({ entries }: { entries: Entry[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || entries.length === 0) return;

    const draw = () => drawScatter(canvas, entries);
    draw();

    const obs = new ResizeObserver(draw);
    obs.observe(canvas);
    return () => obs.disconnect();
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
