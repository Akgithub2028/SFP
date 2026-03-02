"""
scripts/modal_bench.py — Run the leaderboard benchmark on Modal (GPU).

Presets (pick one):
    modal run scripts/modal_bench.py --preset smoke     # ~$0.50, 2 min  — sanity check
    modal run scripts/modal_bench.py --preset quick      # ~$2,   10 min — iterate fast
    modal run scripts/modal_bench.py --preset standard   # ~$8,   45 min — official benchmark
    modal run scripts/modal_bench.py --preset full        # ~$15,  90 min — publication-grade

Custom:
    modal run scripts/modal_bench.py --method sfp --memory 128 --preset quick
"""

from __future__ import annotations

from pathlib import Path

import modal

ROOT_PATH = Path(__file__).parent.parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.40",
        "peft>=0.10",
        "datasets>=2.18",
        "accelerate>=0.28",
        "numpy>=1.24",
        "tqdm>=4.66",
        "pyyaml>=6.0",
        "scipy",
        "prometheus-client>=0.20",
    )
    .add_local_dir(ROOT_PATH, remote_path="/root/sfp", condition=lambda pth: (
        not pth.startswith(".git/")
        and not pth.startswith(".venv/")
        and not pth.startswith("__pycache__/")
        and not pth.startswith("paper/")
        and not pth.startswith("runs/")
        and not pth.startswith(".ruff_cache/")
        and not pth.startswith(".pytest_cache/")
        and not pth.endswith(".pdf")
    ))
)

app = modal.App("sfp-benchmark", image=image)

PUSHGATEWAY = "57.129.90.22:8080"

# ---------------------------------------------------------------------------
# Presets: name -> (gpu, seeds, steps_per_task, model, tasks, est_cost)
# Costs estimated at: T4 ~$0.59/hr, L4 ~$0.80/hr, A10G ~$1.10/hr
# ---------------------------------------------------------------------------

PRESETS = {
    "smoke": {
        "gpu": "T4",
        "seeds": [42],
        "steps_per_task": 50,
        "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "tasks": "math,code",
        "est_cost": "$0.50",
        "est_time": "~2 min",
        "desc": "Sanity check. Tiny model, 1 seed, 50 steps. Just verifies your method runs.",
    },
    "quick": {
        "gpu": "T4",
        "seeds": [42],
        "steps_per_task": 200,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "tasks": "math,code,ifeval",
        "est_cost": "$2",
        "est_time": "~10 min",
        "desc": "Fast iteration. Real model, 1 seed, 200 steps. Good for development.",
    },
    "standard": {
        "gpu": "A10G",
        "seeds": [42, 43, 44],
        "steps_per_task": 1000,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "tasks": "math,code,ifeval",
        "est_cost": "$8",
        "est_time": "~45 min",
        "desc": "Official leaderboard benchmark. 3 seeds, 1000 steps. Used for PR scoring.",
    },
    "full": {
        "gpu": "A10G",
        "seeds": [42, 43, 44],
        "steps_per_task": 2000,
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "tasks": "math,code,ifeval,safety",
        "est_cost": "$15",
        "est_time": "~90 min",
        "desc": "Extended benchmark. 4 tasks, 2000 steps. For paper-grade results.",
    },
}


@app.function(gpu="T4", timeout=7200)
def run_seed_t4(
    method: str, memory: int, seed: int, steps: int, model_name: str, tasks: str,
) -> dict:
    """Run benchmark on T4."""
    return _run(method, memory, seed, steps, model_name, tasks)


@app.function(gpu="A10G", timeout=7200)
def run_seed_a10g(
    method: str, memory: int, seed: int, steps: int, model_name: str, tasks: str,
) -> dict:
    """Run benchmark on A10G."""
    return _run(method, memory, seed, steps, model_name, tasks)


def _run(method: str, memory: int, seed: int, steps: int, model_name: str, tasks: str) -> dict:
    """Shared training logic. Pushes metrics to Prometheus on dev-georgios."""
    import json
    import subprocess

    out_dir = f"/root/sfp/out/bench/{method}_m{memory}_s{seed}"

    cmd = [
        "python", "train.py",
        "--method", method,
        "--memory", str(memory),
        "--seed", str(seed),
        "--model", model_name,
        "--tasks", tasks,
        "--steps_per_task", str(steps),
        "--lora_rank", "64",
        "--out_dir", out_dir,
        "--mode", "fast",
        "--pushgateway", PUSHGATEWAY,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/root/sfp")
    print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-3000:])
        raise RuntimeError(f"Training failed (seed={seed}): {result.stderr[-500:]}")

    results_path = f"{out_dir}/results.json"
    with open(results_path) as f:
        return json.load(f)


@app.local_entrypoint()
def main(method: str = "naive", memory: int = 128, preset: str = "quick"):
    """Run benchmark with cost-aware presets."""
    import json
    import statistics

    if preset not in PRESETS:
        print(f"Unknown preset: {preset}")
        print(f"Available: {', '.join(PRESETS.keys())}")
        raise SystemExit(1)

    p = PRESETS[preset]
    seeds = p["seeds"]
    steps = p["steps_per_task"]
    model_name = p["model"]
    tasks = p["tasks"]
    gpu = p["gpu"]

    print("=" * 60)
    print("  SFP Leaderboard Benchmark")
    print("=" * 60)
    print(f"  Preset     : {preset} — {p['desc']}")
    print(f"  Est. cost  : {p['est_cost']}")
    print(f"  Est. time  : {p['est_time']}")
    print(f"  GPU        : {gpu}")
    print(f"  Method     : {method}")
    print(f"  Memory     : {memory}")
    print(f"  Model      : {model_name}")
    print(f"  Tasks      : {tasks}")
    print(f"  Steps/task : {steps}")
    print(f"  Seeds      : {seeds}")
    print("=" * 60)
    print()

    # Pick the right GPU function
    run_fn = run_seed_a10g if gpu == "A10G" else run_seed_t4

    # Run all seeds in parallel on Modal
    results = list(run_fn.map(
        [method] * len(seeds),
        [memory] * len(seeds),
        seeds,
        [steps] * len(seeds),
        [model_name] * len(seeds),
        [tasks] * len(seeds),
    ))

    # Aggregate
    retentions = [r.get("retention", 0.0) for r in results]
    plasticities = [r.get("plasticity", 0.0) for r in results]
    scores = [r.get("score", 0.0) for r in results]

    ret_mean = statistics.mean(retentions)
    pla_mean = statistics.mean(plasticities)
    sco_mean = statistics.mean(scores)
    ret_std = statistics.stdev(retentions) if len(retentions) > 1 else 0.0
    pla_std = statistics.stdev(plasticities) if len(plasticities) > 1 else 0.0
    sco_std = statistics.stdev(scores) if len(scores) > 1 else 0.0

    print(f"\n{'=' * 60}")
    print("  RESULTS")
    print(f"{'=' * 60}")
    print(f"  Preset     : {preset}")
    print(f"  Method     : {method}")
    print(f"  Memory     : {memory}")
    print(f"  Retention  : {ret_mean:.4f} +/- {ret_std:.4f}")
    print(f"  Plasticity : {pla_mean:.4f} +/- {pla_std:.4f}")
    print(f"  Score      : {sco_mean:.4f} +/- {sco_std:.4f}")
    if preset != "standard":
        print(f"\n  NOTE: This is a '{preset}' run. For official scores, use --preset standard")
    print(f"{'=' * 60}\n")

    # Write machine-readable output
    output = {
        "method": method,
        "memory": memory,
        "preset": preset,
        "gpu": gpu,
        "retention_mean": round(ret_mean, 4),
        "retention_std": round(ret_std, 4),
        "plasticity_mean": round(pla_mean, 4),
        "plasticity_std": round(pla_std, 4),
        "score_mean": round(sco_mean, 4),
        "score_std": round(sco_std, 4),
        "per_seed": results,
    }
    out_path = Path("benchmark_results.json")
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"Results written to {out_path}")
