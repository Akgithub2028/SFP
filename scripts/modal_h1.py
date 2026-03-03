"""
scripts/modal_h1.py — Run the H1 control experiment on Modal (GPU).

Usage:
    modal run scripts/modal_h1.py                    # Qwen-1.5B, 500 steps, ~$3, ~20 min
    modal run scripts/modal_h1.py --steps 200        # Shorter run for quick check
    modal run scripts/modal_h1.py --sweep-r           # Include r-sweep diagnostic
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
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
    )
    .add_local_dir(ROOT_PATH, remote_path="/root/sfp", ignore=[
        ".git/", ".venv/", "__pycache__/", "paper/", "runs/",
        ".ruff_cache/", ".pytest_cache/", "*.pdf", "out/",
    ])
)

app = modal.App("sfp-h1-control", image=image)

vol = modal.Volume.from_name("sfp-h1-results", create_if_missing=True)


@app.function(gpu="A10G", timeout=7200, volumes={"/results": vol})
def run_h1(
    model: str,
    steps: int,
    every: int,
    pca_k: int,
    preserve_r: int,
    memory: int,
    sweep_r: bool,
) -> str:
    """Run H1 control experiment on GPU."""
    import subprocess

    out_dir = "/results/h1_control_gpu"

    cmd = [
        "python", "scripts/h1_control.py",
        "--model", model,
        "--steps", str(steps),
        "--every", str(every),
        "--pca_k", str(pca_k),
        "--preserve_r", str(preserve_r),
        "--memory", str(memory),
        "--out_dir", out_dir,
        "--batch_size", "4",
        "--lr", "1e-4",
        "--lora_rank", "32",
    ]
    if sweep_r:
        cmd.append("--sweep_r")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/root/sfp")
    output = result.stdout
    if result.returncode != 0:
        output += "\n\nSTDERR:\n" + result.stderr
        print(output)
        raise RuntimeError(f"H1 control failed: {result.stderr[-500:]}")

    print(output)

    # Commit volume so results persist
    vol.commit()

    return output


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    steps: int = 500,
    every: int = 25,
    pca_k: int = 128,
    preserve_r: int = 32,
    memory: int = 128,
    sweep_r: bool = True,
):
    """Run the definitive H1 control test on GPU."""
    print("=" * 60)
    print("  H1 Control — GPU Retest (Definitive)")
    print("=" * 60)
    print(f"  Model      : {model}")
    print(f"  Steps      : {steps} (checkpoint every {every})")
    print(f"  PCA k      : {pca_k}, preserve_r: {preserve_r}")
    print(f"  Memory     : {memory}")
    print(f"  Sweep r    : {sweep_r}")
    print(f"  GPU        : A10G")
    print(f"  Est. time  : ~20 min")
    print(f"  Est. cost  : ~$3")
    print("=" * 60)
    print()

    output = run_h1.remote(
        model=model,
        steps=steps,
        every=every,
        pca_k=pca_k,
        preserve_r=preserve_r,
        memory=memory,
        sweep_r=sweep_r,
    )

    # Download results from volume
    import os
    local_out = os.path.join(str(ROOT_PATH), "out", "h1_control_gpu")
    os.makedirs(local_out, exist_ok=True)

    print(f"\nResults available in Modal volume 'sfp-h1-results'")
    print(f"To download: modal volume get sfp-h1-results h1_control_gpu/ {local_out}/")
