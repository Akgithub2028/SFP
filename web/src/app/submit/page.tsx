import Nav from "@/components/Nav";
import Footer from "@/components/Footer";

function Code({ children }: { children: string }) {
  return (
    <pre
      className="overflow-x-auto rounded-md p-4 text-sm leading-relaxed"
      style={{ background: "var(--bg)", border: "1px solid var(--border)" }}
    >
      <code>{children}</code>
    </pre>
  );
}

export default function SubmitPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-6 sm:px-6">
      <Nav />

      <h2 className="mb-6 text-2xl font-bold">Submit Your Method</h2>

      <div className="space-y-8 text-sm leading-relaxed" style={{ color: "var(--text-dim)" }}>
        {/* Overview */}
        <section>
          <p>
            Write a loss function. We run it on a cloud GPU. Your score appears on the leaderboard.{" "}
            <span style={{ color: "var(--text)" }}>No GPU needed on your end.</span>
          </p>
        </section>

        {/* Step 1 */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            <span style={{ color: "var(--accent)" }}>1.</span> Write your method
          </h3>
          <p className="mb-3">
            Copy the example submission and implement a function ending in <code className="px-1" style={{ background: "var(--surface)" }}>_loss</code>.
            Your function receives the current batch, an optional memory batch from previous tasks, and any state
            from your setup function. Return a scalar loss tensor.
          </p>
          <Code>{`cp submissions/_example.py submissions/my_method.py`}</Code>
          <p className="mt-3 mb-3">Here&apos;s the simplest possible method — experience replay:</p>
          <Code>{`import torch
from torch import Tensor

def my_replay_loss(
    model,
    batch: dict,
    memory_batch: dict | None = None,
    replay_ratio: float = 0.2,
    **kw,
) -> Tensor:
    """Mix new-task CE with memory replay."""
    out = model(**batch)
    if memory_batch is None:
        return out.loss
    mem_out = model(**memory_batch)
    return (1 - replay_ratio) * out.loss + replay_ratio * mem_out.loss`}</Code>
          <p className="mt-3">
            Your function can use anything from PyTorch. You can also define a <code className="px-1" style={{ background: "var(--surface)" }}>SETUP</code> attribute
            to run code at each task boundary (e.g., snapshot a teacher model for distillation).
            See <a href="https://github.com/paradigmxyz/sfp/blob/master/methods.py" style={{ color: "var(--accent)" }}>methods.py</a> for
            all baseline implementations.
          </p>
        </section>

        {/* Step 2 */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            <span style={{ color: "var(--accent)" }}>2.</span> Submit
          </h3>
          <Code>{`python submit.py submissions/my_method.py`}</Code>
          <p className="mt-3">
            This uploads your code to the benchmark server, which validates it (syntax, blocked imports)
            and spins up a Modal GPU. The benchmark runs 3 seeds × 1000 steps per task on an A10G.
            Takes about 45 minutes. The CLI shows a progress spinner while it runs.
          </p>
        </section>

        {/* Step 3 */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            <span style={{ color: "var(--accent)" }}>3.</span> Score
          </h3>
          <p>
            When the benchmark finishes, your score is computed and added to the leaderboard automatically.
            The CLI prints your final score. You can also check the{" "}
            <a href="/" style={{ color: "var(--accent)" }}>leaderboard page</a> directly.
          </p>
        </section>

        {/* Local testing */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Test locally first (optional)
          </h3>
          <p className="mb-3">Quick sanity check on CPU (~5 min):</p>
          <Code>{`python train.py --method my_replay \\
    --method_file submissions/my_method.py \\
    --memory 128 --steps_per_task 50 \\
    --model HuggingFaceTB/SmolLM2-135M-Instruct \\
    --tasks math,code --mode smoke`}</Code>
        </section>

        {/* What's fixed */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Rules
          </h3>
          <div
            className="rounded-md p-4"
            style={{ background: "var(--surface)", border: "1px solid var(--border)" }}
          >
            <div className="grid gap-2 sm:grid-cols-2">
              {[
                ["Model", "Qwen2.5-1.5B-Instruct"],
                ["Tasks", "math → code → ifeval"],
                ["Steps", "1000 per task"],
                ["LoRA rank", "64"],
                ["Seeds", "42, 43, 44 (mean ± std)"],
                ["GPU", "A10G (Modal)"],
                ["Memory tiers", "M=128, M=512"],
              ].map(([k, v]) => (
                <div key={k}>
                  <span style={{ color: "var(--text-dim)" }}>{k}: </span>
                  <span className="font-semibold" style={{ color: "var(--text)" }}>{v}</span>
                </div>
              ))}
            </div>
            <ul className="mt-4 list-inside list-disc space-y-1" style={{ color: "var(--text-dim)" }}>
              <li>Same total tokens as baselines (enforced by harness)</li>
              <li>Same memory buffer (provided, not custom)</li>
              <li>No access to eval data during training</li>
              <li>Must be reproducible across 3 seeds</li>
              <li>
                No changes to <code className="px-1" style={{ background: "var(--bg)" }}>train.py</code>,{" "}
                <code className="px-1" style={{ background: "var(--bg)" }}>evaluate.py</code>,{" "}
                <code className="px-1" style={{ background: "var(--bg)" }}>data.py</code>, or{" "}
                <code className="px-1" style={{ background: "var(--bg)" }}>model.py</code>
              </li>
            </ul>
          </div>
        </section>
      </div>

      <Footer />
    </div>
  );
}
