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
    <div className="mx-auto max-w-7xl px-4 py-6 sm:px-6 lg:px-8">
      <Nav />

      <h2 className="mb-2 text-2xl font-bold">Motivation</h2>
      <p className="mb-8 text-sm" style={{ color: "var(--text-dim)" }}>
        Problem setting, prior work, hypotheses, and known limitations.
      </p>

      <article className="space-y-8 text-sm leading-relaxed" style={{ color: "var(--text-dim)" }}>
        {/* The problem */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Problem setting
          </h3>
          <p>
            Large language models are updated sequentially — alignment tuning, safety patches, domain
            adaptation, personalization — and each round of updates can degrade performance on previously
            validated behaviors<Ref id="1" />. In the continual learning literature, this is known as
            catastrophic forgetting<Ref id="2" /><Ref id="3" />.
          </p>
          <p className="mt-3">
            This is distinct from the batch-retraining paradigm used in most production LLM pipelines. We
            focus on the setting where a model is updated <em>in place</em> over a sequence of tasks —
            relevant for personalization, on-device adaptation, and rapid iteration cycles where full
            retraining is too expensive<Ref id="4" />.
          </p>
          <p className="mt-3">
            The phenomenon is straightforward to reproduce: fine-tune a model that can solve math problems
            on a code generation task for a few hundred gradient steps, then re-evaluate on math. Performance
            drops measurably.
          </p>
          <div className="mt-3">
            <CodeBlock language="bash">{`python forget.py`}</CodeBlock>
          </div>
        </section>

        {/* Why it happens */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Mechanism
          </h3>
          <p>
            Neural networks store knowledge in distributed weight patterns with no task-specific
            compartmentalization<Ref id="2" />. When weights are updated for a new objective, nothing
            constrains those updates to preserve previously learned functions. If the loss gradients for
            old and new tasks conflict (∇L<sub>A</sub> · ∇L<sub>B</sub> &lt; 0), old-task performance
            degrades<Ref id="5" />.
          </p>
          <p className="mt-3">
            Under parameter-efficient fine-tuning with low-rank adapters<Ref id="6" />, sequential tasks
            compete for a shared low-rank subspace of the weight space, which can amplify interference.
            However, this also means the space of possible parameter changes is low-dimensional by
            construction — a confound that must be accounted for when studying the dimensionality of
            forgetting.
          </p>
        </section>

        {/* Prior work */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Prior work
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
                  <td>Important weights (Fisher diagonal)</td>
                  <td>Weight-space regularization; agnostic to what weights encode</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text)" }}>LwF<Ref id="7" /></td>
                  <td>Output logits via knowledge distillation</td>
                  <td>Preserves input-output mapping, not internal representations</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text)" }}>PODNet<Ref id="8" /></td>
                  <td>Pooled hidden states at all dimensions</td>
                  <td>Preserves all representational capacity; reduces plasticity</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text)" }}>Orthogonal LoRA</td>
                  <td>Parameter-space directions via orthogonality constraints</td>
                  <td>Operates in weight space, not activation/feature space</td>
                </tr>
                <tr>
                  <td style={{ color: "var(--text)" }}>Experience replay<Ref id="9" /></td>
                  <td>Data samples from previous tasks</td>
                  <td>Requires data storage; effectiveness depends on buffer size and sampling</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-3">
            These approaches make different bets about what to preserve: weights, logits, hidden states,
            parameter directions, or raw data. A common limitation is that none directly identify <em>which
            internal representations</em> are causally responsible for old-task performance.
          </p>
        </section>

        {/* Hypotheses */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Hypotheses
          </h3>
          <p>
            We investigate whether forgetting during continual fine-tuning is predictable from
            low-dimensional structure in the activation space. Specifically:
          </p>
          <div className="mt-3 space-y-3">
            <div className="rounded-md p-3" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="font-semibold" style={{ color: "var(--text)" }}>
                H1: Regressions are predictable from low-dimensional activation drift
              </div>
              <div className="mt-1">
                Drift in the top-<em>r</em> PCA dimensions of mid-layer activations predicts old-task
                accuracy drops (R² ≥ 0.6), while drift in random-<em>r</em> dimensions does not — even
                after controlling for total activation drift ‖Δa‖. This is a predictive correlation claim,
                not a claim about the intrinsic dimensionality of forgetting.
              </div>
            </div>
            <div className="rounded-md p-3" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="font-semibold" style={{ color: "var(--text)" }}>
                H2: Selective preservation improves the retention–plasticity Pareto frontier
              </div>
              <div className="mt-1">
                Preserving only high-importance feature directions (via PCA + ablation ranking) dominates
                baselines that preserve all dimensions<Ref id="8" /> or operate in weight
                space<Ref id="3" />, at small memory budgets (M ≤ 128 examples).
              </div>
            </div>
            <div className="rounded-md p-3" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
              <div className="font-semibold" style={{ color: "var(--text)" }}>
                H3: Preserved features are mechanistically coherent
              </div>
              <div className="mt-1">
                The activation directions identified as important by ablation are interpretable and
                causally linked to task-specific behavior — connecting to recent work on
                mechanistic interpretability of neural networks<Ref id="10" />.
              </div>
            </div>
          </div>
        </section>

        {/* Method */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Method: Sparse Feature Preservation
          </h3>
          <p>
            SFP adds a regularization term that penalizes activation drift in the identified subspace
            during new-task training. The procedure: (1) collect activations on a memory buffer at
            selected layers, (2) compute a PCA basis, (3) rank dimensions by ablation impact on old-task
            performance, (4) select the top-<em>r</em> directions, (5) add a preservation loss
            ‖U<sub>r</sub><sup>⊤</sup> a(θ) − U<sub>r</sub><sup>⊤</sup> a(θ*)‖².
            Implementation is ~40 lines of code.
          </p>
          <p className="mt-3">
            The key distinction from PODNet-style approaches<Ref id="8" /> is selectivity: SFP preserves
            only a small fraction of the representational space (<em>r</em> ≈ 32 out of <em>d</em> ≈ 2048
            per layer), leaving the remaining capacity free for new-task learning. Whether this selectivity
            matters — vs. preserving all dimensions or random dimensions — is tested via ablation controls.
          </p>
          <p className="mt-3">
            The mathematical reduction (H1 + preservation loss ≤ δ → forgetting ≤ C√δ) is verified
            in Lean 4<Ref id="11" />. This makes the algebraic steps machine-checked but does not
            guarantee the empirical premises: whether H1 holds, whether the constant <em>C</em> is
            non-vacuous, or whether SGD achieves small δ in practice.
          </p>
        </section>

        {/* Benchmark */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Benchmark design
          </h3>
          <p>
            A base model is LoRA-fine-tuned<Ref id="6" /> on a sequence of tasks (math → code →
            instruction-following). After all tasks, we measure:
          </p>
          <ul className="mt-3 list-inside list-disc space-y-1">
            <li>
              <strong style={{ color: "var(--text)" }}>Retention</strong> — mean primary metric on previously trained tasks
            </li>
            <li>
              <strong style={{ color: "var(--text)" }}>Plasticity</strong> — primary metric on the most recently trained task
            </li>
            <li>
              <strong style={{ color: "var(--text)" }}>Standard CL metrics</strong> — Average Accuracy,
              Backward Transfer<Ref id="5" />, Forward Transfer, per-task forgetting
            </li>
          </ul>
          <p className="mt-3">
            All other variables are controlled: model architecture, task data, training steps, LoRA
            configuration, memory budget. The composite leaderboard score (0.6·retention + 0.4·plasticity)
            is a convenience ranking; full Pareto frontiers and per-task breakdowns are reported for
            scientific evaluation.
          </p>
        </section>

        {/* Experimental status */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Experimental results
          </h3>
          <p>
            We test H1 with two complementary approaches: correlational analysis (does drift in
            top-<em>r</em> dimensions predict forgetting better than alternatives?) and causal
            intervention (does perturbing top-<em>r</em> dimensions at inference degrade old-task
            performance more than perturbing random dimensions?).
          </p>

          <div className="mt-4 rounded-md p-3" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <div className="font-semibold mb-2" style={{ color: "var(--text)" }}>
              Correlational H1 — Qwen2.5-1.5B-Instruct (Math→Code, 500 steps)
            </div>
            <div className="overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    <th>Predictor</th>
                    <th>Dims</th>
                    <th>R²</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ color: "var(--accent)" }}>Top-<em>r</em> (ablation)</td>
                    <td>32</td>
                    <td style={{ color: "var(--accent)" }}>0.933</td>
                  </tr>
                  <tr>
                    <td>Total drift</td>
                    <td>128</td>
                    <td>0.888</td>
                  </tr>
                  <tr>
                    <td>Random dims</td>
                    <td>32</td>
                    <td>0.850</td>
                  </tr>
                  <tr>
                    <td>Bottom-<em>r</em> (ablation)</td>
                    <td>32</td>
                    <td>0.827</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="mt-2">
              Correct ranking: top {">"} total {">"} random {">"} bottom. H1 partially supported —
              top-<em>r</em> is the best predictor, but the margin over random is modest (0.933 vs 0.850).
              This result fails at 135M scale (ranking inverted), suggesting H1 is scale-dependent.
            </p>
          </div>

          <div className="mt-4 rounded-md p-3" style={{ background: "var(--surface)", border: "1px solid var(--border)" }}>
            <div className="font-semibold mb-2" style={{ color: "var(--text)" }}>
              Causal intervention — SmolLM2-135M (noise injection at inference)
            </div>
            <div className="overflow-x-auto">
              <table>
                <thead>
                  <tr>
                    <th>Intervention</th>
                    <th>Δ perplexity</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td style={{ color: "var(--accent)" }}>Top-<em>r</em></td>
                    <td style={{ color: "var(--accent)" }}>+0.85</td>
                  </tr>
                  <tr>
                    <td>Random</td>
                    <td>+0.21</td>
                  </tr>
                  <tr>
                    <td>Bottom-<em>r</em></td>
                    <td>+0.02</td>
                  </tr>
                </tbody>
              </table>
            </div>
            <p className="mt-2">
              Top-<em>r</em> causes 4× more degradation than random and 42× more than bottom.
              Notably, this passes even at 135M where correlational H1 fails — ablation importance
              correctly identifies causally important dimensions regardless of scale.
            </p>
          </div>
        </section>

        {/* Limitations */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            Threats to validity
          </h3>
          <ul className="mt-1 list-inside list-disc space-y-1">
            <li>
              <strong style={{ color: "var(--text)" }}>LoRA confound.</strong> Low-rank adapters constrain
              updates to a low-dimensional subspace by construction<Ref id="6" />. &ldquo;Low-dimensional
              forgetting&rdquo; may be an artifact of the parameter-efficient training regime, not a
              property of forgetting itself. Full fine-tuning experiments are needed.
            </li>
            <li>
              <strong style={{ color: "var(--text)" }}>Model scale.</strong> Primary experiments use
              135M–1.5B parameter models. Results may not transfer to larger scales (7B+).
            </li>
            <li>
              <strong style={{ color: "var(--text)" }}>Task order sensitivity.</strong> A fixed curriculum
              may favor certain methods. Multiple orderings should be evaluated.
            </li>
            <li>
              <strong style={{ color: "var(--text)" }}>Benchmark coverage.</strong> Five tasks (math, code,
              instruction-following, safety, domain) represent a limited sample of LLM capabilities.
            </li>
            <li>
              <strong style={{ color: "var(--text)" }}>Bound tightness.</strong> The constant <em>C</em> in
              the formal reduction is estimated empirically. If <em>C</em> is large or unstable, the
              bound is vacuous.
            </li>
            <li>
              <strong style={{ color: "var(--text)" }}>Layer selection.</strong> Probing activations at
              pre-selected depths (¼, ½, ¾) risks overfitting the layer choice to the benchmark.
            </li>
          </ul>
        </section>

        {/* References */}
        <section>
          <h3 className="mb-3 text-lg font-semibold" style={{ color: "var(--text)" }}>
            References
          </h3>
          <ol className="list-inside list-decimal space-y-2">
            <Cite
              id="1"
              authors="Luo et al."
              year="2024"
              title="An Empirical Study of Catastrophic Forgetting in Large Language Models During Continual Fine-tuning"
              url="https://arxiv.org/abs/2308.08747"
            />
            <Cite
              id="2"
              authors="McCloskey & Cohen"
              year="1989"
              title="Catastrophic Interference in Connectionist Networks: The Sequential Learning Problem"
            />
            <Cite
              id="3"
              authors="Kirkpatrick et al."
              year="2017"
              title="Overcoming catastrophic forgetting in neural networks"
              url="https://arxiv.org/abs/1612.00796"
            />
            <Cite
              id="4"
              authors="Wang et al."
              year="2024"
              title="A Comprehensive Survey of Continual Learning for Large Language Models"
              url="https://arxiv.org/abs/2404.16789"
            />
            <Cite
              id="5"
              authors="Lopez-Paz & Ranzato"
              year="2017"
              title="Gradient Episodic Memory for Continual Learning"
              url="https://arxiv.org/abs/1706.08840"
            />
            <Cite
              id="6"
              authors="Hu et al."
              year="2022"
              title="LoRA: Low-Rank Adaptation of Large Language Models"
              url="https://arxiv.org/abs/2106.09685"
            />
            <Cite
              id="7"
              authors="Li & Hoiem"
              year="2017"
              title="Learning without Forgetting"
              url="https://arxiv.org/abs/1606.09282"
            />
            <Cite
              id="8"
              authors="Douillard et al."
              year="2020"
              title="PODNet: Pooled Outputs Distillation for Small-Tasks Incremental Learning"
              url="https://arxiv.org/abs/2004.13513"
            />
            <Cite
              id="9"
              authors="Chaudhry et al."
              year="2019"
              title="On Tiny Episodic Memories in Continual Learning"
              url="https://arxiv.org/abs/1902.10486"
            />
            <Cite
              id="10"
              authors="Conmy et al."
              year="2023"
              title="Towards Automated Circuit Discovery for Mechanistic Interpretability"
              url="https://arxiv.org/abs/2304.14997"
            />
            <Cite
              id="11"
              authors="de Moura & Ullrich"
              year="2021"
              title="The Lean 4 Theorem Prover and Programming Language"
              url="https://doi.org/10.1007/978-3-030-79876-5_37"
            />
          </ol>
        </section>
      </article>

      <Footer />
    </div>
  );
}
