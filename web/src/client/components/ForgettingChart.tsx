import { useRef, useEffect } from "react";
import type { DetailEntry } from "@/lib/types";
import { drawForgettingChart } from "@/lib/charts";

export default function ForgettingChart({
  details,
  tasks,
}: {
  details: DetailEntry[];
  tasks: string[];
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || details.length === 0) return;

    const draw = () => drawForgettingChart(canvas, details, tasks);
    draw();

    const obs = new ResizeObserver(draw);
    obs.observe(canvas);
    return () => obs.disconnect();
  }, [details, tasks]);

  if (details.length === 0) {
    return (
      <div className="chart flex items-center justify-center" style={{ color: "var(--text-dim)" }}>
        Run benchmarks to see the forgetting chart.
      </div>
    );
  }

  return <canvas ref={canvasRef} className="chart chart-tall" />;
}
