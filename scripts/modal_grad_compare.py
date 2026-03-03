"""
scripts/modal_grad_compare.py — Run gradient basis vs PCA comparison on Modal (GPU).

Usage:
    modal run scripts/modal_grad_compare.py --detach      # Qwen-1.5B, ~$5, ~30 min
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

app = modal.App("sfp-grad-compare", image=image)

vol = modal.Volume.from_name("sfp-grad-compare-results", create_if_missing=True)


@app.function(gpu="A10G", timeout=7200, volumes={"/results": vol})
def run_grad_compare(
    model: str, steps: int, every: int, memory: int, pca_k: int, preserve_r: int,
) -> str:
    import subprocess

    out_dir = "/results/grad_compare"

    cmd = [
        "python", "scripts/gradient_basis_compare.py",
        "--model", model,
        "--steps", str(steps),
        "--every", str(every),
        "--memory", str(memory),
        "--pca_k", str(pca_k),
        "--preserve_r", str(preserve_r),
        "--out_dir", out_dir,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd="/root/sfp")
    output = result.stdout
    if result.returncode != 0:
        output += "\n\nSTDERR:\n" + result.stderr
        print(output)
        raise RuntimeError(f"Grad compare failed: {result.stderr[-500:]}")

    print(output)
    vol.commit()
    return output


@app.local_entrypoint()
def main(
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    steps: int = 500,
    every: int = 25,
    memory: int = 128,
    pca_k: int = 128,
    preserve_r: int = 32,
):
    print("=" * 60)
    print("  Gradient Basis vs PCA Comparison — Modal GPU")
    print("=" * 60)
    print(f"  Model      : {model}")
    print(f"  Steps      : {steps} (every {every})")
    print(f"  Memory     : {memory}")
    print(f"  PCA k      : {pca_k}, preserve_r: {preserve_r}")
    print(f"  GPU        : A10G")
    print("=" * 60)
    print()

    output = run_grad_compare.remote(
        model=model,
        steps=steps,
        every=every,
        memory=memory,
        pca_k=pca_k,
        preserve_r=preserve_r,
    )

    print(f"\nResults in Modal volume 'sfp-grad-compare-results'")
    print(f"To download: modal volume get sfp-grad-compare-results grad_compare/ out/grad_compare/")
