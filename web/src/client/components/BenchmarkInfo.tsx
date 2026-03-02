import type { Benchmark } from "@/lib/types";

export default function BenchmarkInfo({ benchmark }: { benchmark: Benchmark }) {
  const items = [
    { label: "Model", value: benchmark.model },
    { label: "Tasks", value: benchmark.tasks.join(" → ") },
    { label: "Steps/task", value: benchmark.steps_per_task.toLocaleString() },
    { label: "LoRA rank", value: benchmark.lora_rank },
    { label: "Seeds", value: benchmark.seeds.join(", ") },
    { label: "Memory", value: `${benchmark.memory} tokens` },
  ];

  return (
    <div
      className="grid grid-cols-2 gap-x-8 gap-y-2 rounded-md p-4 text-sm sm:grid-cols-3"
      style={{ border: "1px solid var(--border)", background: "var(--surface)" }}
    >
      {items.map((item) => (
        <div key={item.label}>
          <span style={{ color: "var(--text-dim)" }}>{item.label}: </span>
          <span className="font-semibold">{item.value}</span>
        </div>
      ))}
    </div>
  );
}
