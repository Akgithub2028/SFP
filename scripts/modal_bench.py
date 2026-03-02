"""
scripts/modal_bench.py — Run the leaderboard benchmark on Modal (GPU).

Called from GitHub Actions on PR submissions. Runs the fixed benchmark:
  - Model: Qwen/Qwen2.5-1.5B-Instruct
  - Tasks: math -> code -> ifeval
  - Steps: 1000 per task
  - Seeds: 42, 43, 44
  - Memory: 128

Usage:
    modal run scripts/modal_bench.py --method my_method
    modal run scripts/modal_bench.py --method sfp --memory 512
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

SEEDS = [42, 43, 44]


@app.function(gpu="A10G", timeout=7200)
def run_seed(method: str, memory: int, seed: int) -> dict:
    """Run the benchmark for a single seed. Returns results dict."""
    import json
    import subprocess

    out_dir = f"/root/sfp/out/bench/{method}_m{memory}_s{seed}"

    cmd = [
        "python", "train.py",
        "--method", method,
        "--memory", str(memory),
        "--seed", str(seed),
        "--model", "Qwen/Qwen2.5-1.5B-Instruct",
        "--tasks", "math,code,ifeval",
        "--steps_per_task", "1000",
        "--lora_rank", "64",
        "--out_dir", out_dir,
        "--mode", "fast",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/root/sfp")
    print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:])
        raise RuntimeError(f"Training failed (seed={seed}): {result.stderr[-500:]}")

    results_path = f"{out_dir}/results.json"
    with open(results_path) as f:
        return json.load(f)


@app.local_entrypoint()
def main(method: str = "naive", memory: int = 128):
    """Run benchmark across 3 seeds and aggregate results."""
    import json
    import statistics

    print("=== SFP Leaderboard Benchmark ===")
    print(f"Method: {method}, Memory: {memory}")
    print(f"Seeds: {SEEDS}")
    print()

    # Run all seeds in parallel on Modal
    results = list(run_seed.map(
        [method] * len(SEEDS),
        [memory] * len(SEEDS),
        SEEDS,
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
    print("  LEADERBOARD RESULTS")
    print(f"{'=' * 60}")
    print(f"  Method     : {method}")
    print(f"  Memory     : {memory}")
    print(f"  Retention  : {ret_mean:.4f} +/- {ret_std:.4f}")
    print(f"  Plasticity : {pla_mean:.4f} +/- {pla_std:.4f}")
    print(f"  Score      : {sco_mean:.4f} +/- {sco_std:.4f}")
    print(f"{'=' * 60}\n")

    # Write machine-readable output for CI
    output = {
        "method": method,
        "memory": memory,
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
