import Nav from "@/components/Nav";
import Footer from "@/components/Footer";
import CodeBlock from "@/components/CodeBlock";

function Cite({ id, title, authors, year, url }: {
  id: string; title: string; authors: string; year: string; url?: string;
}) {
  return (
    <li id={id} className="text-xs leading-relaxed" style={{ color: "var(--text-dim)" }}>
      <span style={{ color: "var(--text)" }}>{authors}</span> ({year}).{" "}
      {url ? (
        <a href={url} style={{ color: "var(--accent)" }}>{title}</a>
      ) : (
        <em>{title}</em>
      )}
    </li>
  );
}

function Ref({ id }: { id: string }) {
  return (
    <a href={`#${id}`} className="text-xs align-super" style={{ color: "var(--accent)" }}>
      [{id}]
    </a>
  );
}

export default function About() {
  return (
    <div className="mx-auto max-w-6xl px-4 py-6 sm:px-6">
      <Nav />

      <h2 className="mb-2 text-2xl font-bold">About</h2>
      <p className="mb-8 text-sm" style={{ color: "var(--text-dim)" }}>
        Why catastrophic forgetting matters, what we know, and what&apos;s still open.
      </p>

      <article className="space-y-8 text-sm leading-relaxed" style={{ color: "var(--text-dim)" }}>
        {/* The problem */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            The problem
          </h3>
          <p>
            Large language models are continually fine-tuned: new domains, new safety alignment, new capabilities.
            Each round of fine-tuning improves the model on the new task — but degrades performance on everything
            it learned before. This is <strong style={{ color: "var(--text)" }}>catastrophic forgetting</strong>
            <Ref id="1" />, and it&apos;s one of the core obstacles to deploying LLMs that learn over time.
          </p>
          <p className="mt-3">
            The problem is simple to demonstrate. Take a model that can do math. Fine-tune it on code generation
            for a few hundred steps. Test it on math again — performance drops significantly. The model hasn&apos;t
            just learned code; it&apos;s <em>unlearned</em> math. You can see this happen in under 5 minutes:
          </p>
          <div className="mt-3">
            <CodeBlock language="bash">{`python forget.py`}</CodeBlock>
          </div>
        </section>

        {/* Why it happens */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Why it happens
          </h3>
          <p>
            Neural networks store knowledge in distributed, entangled weight patterns. There are no &ldquo;task slots.&rdquo;
            When you update weights for Task B, nothing constrains those updates to preserve Task A&apos;s learned
            representations. If the gradients conflict (∇L<sub>A</sub> · ∇L<sub>B</sub> &lt; 0), Task A degrades.
          </p>
          <p className="mt-3">
            This is especially acute with LoRA fine-tuning<Ref id="2" />: the low-rank adapter modifies a shared
            subspace of the model&apos;s weights, and sequential tasks compete for the same limited capacity.
          </p>
        </section>

        {/* What's been tried */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Existing approaches
          </h3>
          <div className="overflow-x-auto">
            <table>
              <thead>
                <tr>
                  <th>Method</th>
                  <th>What it preserves</th>
                  <th>Limitation</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style={{ color: "var(--text)" }}>EWC<Ref id="3" /></td>
                  <td>Important weights (Fisher information)</td>
                  <td>Global; doesn&apos;t know <em>what</em> weights encode</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text)" }}>LwF<Ref id="4" /></td>
                  <td>Output logits of a teacher</td>
                  <td>Doesn&apos;t preserve internal representations</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text)" }}>PODNet<Ref id="5" /></td>
                  <td>Hidden states (all dims)</td>
                  <td>Preserves too much; hurts plasticity</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text)" }}>Orthogonal LoRA</td>
                  <td>Parameter space directions</td>
                  <td>Geometry-based, not feature-based</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text)" }}>Experience replay</td>
                  <td>Data samples from old tasks</td>
                  <td>Relies on data availability and storage</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-3">
            Each approach makes a different bet on <em>what to preserve</em>. Weights, logits, hidden states,
            parameter directions, or raw data. None of them ask the deeper question:
          </p>
          <p className="mt-2 font-semibold" style={{ color: "var(--text)" }}>
            Where inside the model does forgetting actually happen?
          </p>
        </section>

        {/* The open question */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            The open question
          </h3>
          <p>
            Is forgetting spread uniformly across the model&apos;s internal representations — or is it
            concentrated in a low-dimensional subspace?
          </p>
          <p className="mt-3">
            If forgetting is <strong style={{ color: "var(--text)" }}>low-dimensional</strong>, then we
            don&apos;t need to constrain the entire model. We only need to preserve the small number of
            feature directions that actually encode previously-learned capabilities. This would resolve the
            stability-plasticity tradeoff: lock down what matters, let everything else adapt freely.
          </p>
          <p className="mt-3">
            Recent work at the intersection of continual learning and mechanistic interpretability<Ref id="6" />
            suggests this might be the case — that knowledge in neural networks occupies structured, low-rank
            subspaces that can be identified and protected. But this hasn&apos;t been demonstrated convincingly
            for LLM fine-tuning at scale.
          </p>
        </section>

        {/* Our hypothesis */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Our hypothesis
          </h3>
          <p>We conjecture three things:</p>
          <div className="mt-3 space-y-3">
            <div className="rounded-md p-3" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="font-semibold" style={{ color: "var(--text)" }}>
                H1: Forgetting is low-dimensional
              </div>
              <div className="mt-1">
                Drift in the top-r principal components of a layer&apos;s activation space predicts forgetting
                (R² ≥ 0.6). Drift in random-r directions does not.
              </div>
            </div>
            <div className="rounded-md p-3" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="font-semibold" style={{ color: "var(--text)" }}>
                H2: Sparse preservation improves the Pareto frontier
              </div>
              <div className="mt-1">
                Preserving only high-importance features (via PCA + ablation ranking) dominates
                baselines on the retention-plasticity tradeoff at small memory budgets.
              </div>
            </div>
            <div className="rounded-md p-3" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="font-semibold" style={{ color: "var(--text)" }}>
                H3: Preserved features are mechanistically coherent
              </div>
              <div className="mt-1">
                The features identified as important are interpretable and causally linked to
                task-specific behavior — not arbitrary directions.
              </div>
            </div>
          </div>
          <p className="mt-3">
            <strong style={{ color: "var(--text)" }}>Sparse Feature Preservation (SFP)</strong> is our method
            that tests these hypotheses. It hooks activations at mid-layers, builds a PCA basis from memory
            set activations, ranks dimensions by ablation importance, and adds a preservation loss on the
            top-r directions. About 40 lines of code. It&apos;s one entry on the leaderboard — we want to see
            if something better exists.
          </p>
        </section>

        {/* What this benchmark measures */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            What this benchmark measures
          </h3>
          <p>
            The benchmark is deliberately simple. A base model is LoRA-fine-tuned on a sequence of tasks
            (math → code → instruction-following). After all tasks, we measure:
          </p>
          <ul className="mt-3 list-inside list-disc space-y-1">
            <li>
              <strong style={{ color: "var(--text)" }}>Retention</strong> — mean accuracy on old tasks (did the model forget?)
            </li>
            <li>
              <strong style={{ color: "var(--text)" }}>Plasticity</strong> — accuracy on the final task (can it still learn?)
            </li>
          </ul>
          <p className="mt-3">
            Everything else is fixed: model, task order, number of steps, LoRA config, memory budget.
            The only variable is your loss function. This isolates the question: given the same resources,
            what&apos;s the best strategy for preventing forgetting?
          </p>
        </section>

        {/* What this doesn't test */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Known limitations
          </h3>
          <p>
            This benchmark tests a narrow slice of continual learning. It does <em>not</em> cover:
          </p>
          <ul className="mt-3 list-inside list-disc space-y-1">
            <li>Long task horizons (hundreds of sequential updates)</li>
            <li>Continuous distribution shift (real-world data drift)</li>
            <li>Full-weight fine-tuning (LoRA constrains the problem)</li>
            <li>Scale (does the approach hold at 7B, 70B?)</li>
            <li>Task order sensitivity</li>
            <li>Pretraining knowledge preservation</li>
          </ul>
          <p className="mt-3">
            We view this as a starting point. If a method dominates here, the next question is whether
            it generalizes beyond these constraints.
          </p>
        </section>

        {/* References */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            References
          </h3>
          <ol className="list-inside list-decimal space-y-2">
            <Cite
              id="1"
              authors="McCloskey & Cohen"
              year="1989"
              title="Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem"
            />
            <Cite
              id="2"
              authors="Hu et al."
              year="2022"
              title="LoRA: Low-Rank Adaptation of Large Language Models"
              url="https://arxiv.org/abs/2106.09685"
            />
            <Cite
              id="3"
              authors="Kirkpatrick et al."
              year="2017"
              title="Overcoming catastrophic forgetting in neural networks (EWC)"
              url="https://arxiv.org/abs/1612.00796"
            />
            <Cite
              id="4"
              authors="Li & Hoiem"
              year="2017"
              title="Learning without Forgetting (LwF)"
              url="https://arxiv.org/abs/1606.09282"
            />
            <Cite
              id="5"
              authors="Douillard et al."
              year="2020"
              title="PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning"
              url="https://arxiv.org/abs/2004.13513"
            />
            <Cite
              id="6"
              authors="Wang et al."
              year="2024"
              title="A Comprehensive Survey of Continual Learning for Large Language Models"
              url="https://arxiv.org/abs/2404.16789"
            />
            <Cite
              id="7"
              authors="Zheng et al."
              year="2024"
              title="LiveBench: A Challenging, Contamination-Free LLM Benchmark"
              url="https://arxiv.org/abs/2406.19314"
            />
            <Cite
              id="8"
              authors="Zhou et al."
              year="2023"
              title="Instruction-Following Evaluation for Large Language Models (IFEval)"
              url="https://arxiv.org/abs/2311.07911"
            />
          </ol>
        </section>
      </article>

      <Footer />
    </div>
  );
}
