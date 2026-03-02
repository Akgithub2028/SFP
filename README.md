# sfp

**Does your LLM forget? Find out in 5 minutes.**

sfp is the simplest experimental harness for studying catastrophic forgetting in LLMs. Fine-tune a model on sequential tasks, watch it forget, and try methods to prevent it. The code is minimal, hackable, and covers continual training, evaluation, drift analysis, and a forgetting leaderboard. For example, you can watch a small LLM lose its math abilities after learning to code in under 5 minutes on a CPU:

```
python forget.py
```

More generally, sfp lets you run any continual learning method through a standardized benchmark with one command. The main dial is `--memory`: how many examples from previous tasks you're allowed to remember (0 = no memory = maximum forgetting, 2048 = lots of replay). All methods, baselines, and our proposed method (Sparse Feature Preservation) compete on the same footing.

## Quick start

```bash
git clone https://github.com/paradigmxyz/sfp.git && cd sfp
uv sync
python forget.py
```

**I have a GPU.** Great, run the full benchmark:
```bash
python train.py --method sfp --memory 128
```

**I only have a CPU.** No worries, `forget.py` runs fine on CPU in ~5 minutes. You'll see forgetting happen before your eyes.

## Forgetting Leaderboard (M=128)

Who can prevent the most forgetting? Submit your method and find out.

| # | Score↑ | Retention↑ | Plasticity↑ | Method | Description | Date | Commit | Contributor |
|---|--------|------------|-------------|--------|-------------|------|--------|-------------|
| — | — | — | — | — | *your method here* | — | — | — |

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for how to submit.

## What is this?

When you fine-tune an LLM on Task A (say, math) and then fine-tune it on Task B (say, code), it *forgets* how to do math. This is called **catastrophic forgetting**, and it's one of the biggest unsolved problems in deploying LLMs that need to learn new things over time.

sfp provides: (1) a standardized benchmark to measure forgetting across sequential tasks, (2) implementations of common mitigation methods (replay, distillation, orthogonal adapters), and (3) our proposed method — **Sparse Feature Preservation (SFP)** — which identifies a small set of important internal features and preserves only those during new task training.

The key insight behind SFP: forgetting isn't uniformly distributed across the model's representations. It's concentrated in a **low-dimensional feature subspace**. Preserve those ~32 dimensions per layer, and you keep old abilities without sacrificing new ones.

## Getting started

### Watch forgetting happen (5 min, CPU)

```bash
python forget.py
```

This loads a tiny LLM, tests it on math, fine-tunes it on code, and tests math again. You'll see the model lose its math abilities in real time.

### Run a method (1 GPU)

```bash
python train.py --method sfp --memory 128 --model Qwen/Qwen2.5-1.5B-Instruct
```

### Full experiment suite (8×H100)

```bash
bash runs/full.sh
```

## Research

If you're a researcher and want to iterate fast, train on the demo config (~5 min):

```bash
python train.py --config configs/demo.yaml
```

Change something in `methods.py`, re-run, see if it helped. To compare methods:

```bash
python analyze.py --compare out/method_a out/method_b
```

This generates Pareto frontier plots, drift-vs-forgetting regressions, and layer heatmaps.

For the full research context (hypotheses, prior work, experimental design), see [dev/RESEARCH.md](dev/RESEARCH.md).

## File structure

```
.
├── README.md           # You are here
├── forget.py           # 5-minute demo: watch forgetting happen
├── train.py            # Continual training loop (~300 lines)
├── evaluate.py         # Eval harness + forgetting metrics
├── analyze.py          # Drift computation, regression, plots
├── methods.py          # All methods: naive, replay, distill, sfp
├── data.py             # Dataset loading + memory buffers
├── model.py            # Model loading + LoRA setup
├── features.py         # Activation hooks, PCA subspace, importance
├── configs/
│   ├── demo.yaml       # 5-minute demo config
│   ├── fast.yaml       # 1-GPU quick validation
│   └── full.yaml       # 8×H100 NeurIPS suite
├── runs/
│   ├── demo.sh         # Smoke test
│   ├── speedrun.sh     # Fast falsification (Gate 1 + Gate 2)
│   ├── full.sh         # Full experiment suite
│   └── leaderboard.sh  # Run the leaderboard benchmark
├── tasks/
│   ├── math.py         # GSM8K evaluation
│   ├── code.py         # HumanEval evaluation
│   ├── ifeval.py       # Instruction-following evaluation
│   ├── safety.py       # Safety / refusal evaluation
│   └── domain.py       # Domain shift evaluation
├── dev/
│   ├── LEADERBOARD.md  # How to submit to the leaderboard
│   └── RESEARCH.md     # Full research briefing
├── pyproject.toml
└── tests/
```

## Contributing

The goal of sfp is to be the simplest, most hackable benchmark for continual LLM fine-tuning. There are no giant configuration objects, no model factories, no if-then-else monsters. It's a single, cohesive, minimal codebase.

To add a new method:
1. Add a function to `methods.py` (or a new file)
2. Run `bash runs/leaderboard.sh your_method 128`
3. Open a PR with your method + results

See [dev/LEADERBOARD.md](dev/LEADERBOARD.md) for details.

## Cite

```bibtex
@misc{sfp,
  author = {Paradigm},
  title = {sfp: Does your LLM forget? Find out in 5 minutes.},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/paradigmxyz/sfp}
}
```

## License

MIT
