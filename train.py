"""
train.py — Continual fine-tuning training loop.

~300 lines. THE main file. Readable top-to-bottom like nanoGPT's train.py.

Usage:
    python train.py                                    # naive, tiny model, 2 tasks
    python train.py --method replay --memory 128       # replay buffer
    python train.py --method sfp --memory 128          # our method
    python train.py --config configs/fast.yaml         # config-driven
    torchrun --nproc_per_node=8 train.py --config configs/full.yaml  # 8xH100
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from tqdm import tqdm

import data
from evaluate import (
    compute_bwt,
    compute_forgetting,
    compute_retention_plasticity,
    evaluate_all,
    leaderboard_score,
    print_results_table,
)
from methods import METHODS, method_setup
from model import load_model, save_checkpoint

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

config = {
    "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
    "method": "naive",
    "tasks": "math,code",
    "steps_per_task": 200,
    "batch_size": 4,
    "lr": 1e-4,
    "lora_rank": 64,
    "lora_alpha": 128,
    "memory": 0,
    "seed": 42,
    "out_dir": "out/default",
    "mode": "fast",
    "pca_k": 128,
    "preserve_r": 32,
    "sfp_lambda": 0.1,
    "max_length": 512,
    "pushgateway": "",
    "eval_every": 0,
    "method_file": "",
}

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def parse_config() -> dict:
    """Merge defaults ← YAML file ← CLI args. Returns final config dict."""
    parser = argparse.ArgumentParser(description="SFP continual fine-tuning")
    parser.add_argument("--config", type=str, default="", help="YAML config file")
    for key, default in config.items():
        arg_type = type(default) if not isinstance(default, bool) else int
        parser.add_argument(f"--{key}", type=arg_type, default=None)
    args = parser.parse_args()

    cfg = dict(config)

    # YAML overrides
    if args.config:
        import yaml

        with open(args.config) as f:
            yaml_cfg = yaml.safe_load(f) or {}
        cfg.update(yaml_cfg)

    # CLI overrides
    for key in config:
        cli_val = getattr(args, key, None)
        if cli_val is not None:
            cfg[key] = cli_val

    return cfg


def load_method_file(path: str, method_name: str) -> None:
    """Dynamically load a method from an external .py file into METHODS."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("submitted_method", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # Find the loss function (ending in _loss)
    loss_fn = None
    for name in dir(mod):
        obj = getattr(mod, name)
        if callable(obj) and name.endswith("_loss"):
            loss_fn = obj
            break

    if loss_fn is None:
        raise ValueError(f"No function ending in '_loss' found in {path}")

    # Apply SETUP attribute if declared in the module
    setup_key = getattr(mod, "SETUP", None)
    if setup_key:
        loss_fn.SETUP = setup_key

    METHODS[method_name] = loss_fn
    print(f"Loaded method '{method_name}' from {path} (fn={loss_fn.__name__}, SETUP={getattr(loss_fn, 'SETUP', 'none')})")


def print_banner(cfg: dict) -> None:
    """Print a summary banner at start."""
    print("=" * 60)
    print("  SFP — Sparse Feature Preservation")
    print("=" * 60)
    print(f"  model   : {cfg['model']}")
    print(f"  method  : {cfg['method']}")
    print(f"  tasks   : {cfg['tasks']}")
    print(f"  memory  : {cfg['memory']}")
    print(f"  steps   : {cfg['steps_per_task']} per task")
    print(f"  lr      : {cfg['lr']}")
    print(f"  lora    : r={cfg['lora_rank']} alpha={cfg['lora_alpha']}")
    print(f"  seed    : {cfg['seed']}")
    print(f"  out_dir : {cfg['out_dir']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Metrics (optional Prometheus push)
# ---------------------------------------------------------------------------


def metrics_init(cfg: dict):
    """Try to init Prometheus metrics logger. Returns logger or None."""
    if not cfg["pushgateway"]:
        return None
    try:
        from scripts.metrics import MetricsLogger

        return MetricsLogger(
            method=cfg["method"],
            memory=str(cfg["memory"]),
            seed=cfg["seed"],
            model=cfg["model"],
            gateway=cfg["pushgateway"],
        )
    except Exception as e:
        print(f"[warn] metrics init failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = parse_config()

    # Load external method file if specified
    if cfg["method_file"]:
        load_method_file(cfg["method_file"], cfg["method"])

    print_banner(cfg)

    # Seeds
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    # Print full config
    print("\nFull config:")
    for k, v in sorted(cfg.items()):
        print(f"  {k}: {v}")
    print()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model(
        name=cfg["model"],
        lora_rank=cfg["lora_rank"],
        lora_alpha=cfg["lora_alpha"],
    )
    model.train()
    model.print_trainable_parameters()

    # Optimizer — only trainable LoRA params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg["lr"])

    # Parse tasks
    tasks = [t.strip() for t in cfg["tasks"].split(",")]
    print(f"\nTask sequence: {tasks}\n")

    # Metrics (Prometheus pushgateway, optional)
    metrics = metrics_init(cfg)

    # State
    os.makedirs(cfg["out_dir"], exist_ok=True)
    memory_buffers: dict[str, list[dict]] = {}
    results_history: list[dict] = []
    t0 = time.time()

    # -----------------------------------------------------------------------
    # Continual training loop
    # -----------------------------------------------------------------------

    for task_idx, task_name in enumerate(tasks):
        print(f"\n{'=' * 60}")
        print(f"=== Task {task_idx}: {task_name} ===")
        print(f"{'=' * 60}\n")

        # Load training data
        dataset = data.load_task(task_name, split="train")
        print(f"Loaded {len(dataset)} training examples for '{task_name}'")

        # Create memory buffer for this task
        if cfg["memory"] > 0:
            buf = data.create_memory_buffer(dataset, size=cfg["memory"], seed=cfg["seed"])
            memory_buffers[task_name] = buf
            data.save_memory(buf, os.path.join(cfg["out_dir"], f"memory_{task_name}.jsonl"))

        # Build combined memory from all *previous* task buffers, capped at M total
        combined_memory: list[dict] = []
        prev_buffers = {t: memory_buffers[t] for t in tasks[:task_idx] if t in memory_buffers}
        if prev_buffers and cfg["memory"] > 0:
            per_task = max(1, cfg["memory"] // len(prev_buffers))
            for t in tasks[:task_idx]:
                if t in prev_buffers:
                    combined_memory.extend(prev_buffers[t][:per_task])
            # Final cap in case of rounding
            combined_memory = combined_memory[:cfg["memory"]]

        # Method-specific setup at task boundary
        method_state = method_setup(
            cfg["method"],
            model,
            tokenizer,
            memory_buffers,
            k=cfg["pca_k"],
            r=cfg["preserve_r"],
            lam=cfg["sfp_lambda"],
        )

        # Get loss function
        loss_fn = METHODS[cfg["method"]]

        # -------------------------------------------------------------------
        # Training step loop
        # -------------------------------------------------------------------

        pbar = tqdm(range(cfg["steps_per_task"]), desc=f"Task {task_idx}: {task_name}")
        for step in pbar:
            # Current task batch
            batch = data.get_batch(
                dataset,
                batch_size=cfg["batch_size"],
                tokenizer=tokenizer,
                max_length=cfg["max_length"],
            )

            # Memory batch (if memory > 0 and we have previous memory)
            memory_batch = None
            if cfg["memory"] > 0 and combined_memory:
                memory_batch = data.get_batch(
                    combined_memory,
                    batch_size=cfg["batch_size"],
                    tokenizer=tokenizer,
                    max_length=cfg["max_length"],
                )

            # Forward + loss
            loss = loss_fn(
                model,
                batch,
                memory_batch=memory_batch,
                lam=cfg["sfp_lambda"],
                **method_state,
            )

            # Backward
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            loss_val = loss.item()
            if step % 50 == 0:
                elapsed = time.time() - t0
                print(
                    f"  step {step:4d}/{cfg['steps_per_task']} | "
                    f"loss {loss_val:.4f} | "
                    f"elapsed {elapsed:.1f}s"
                )
            pbar.set_postfix(loss=f"{loss_val:.4f}")
            if metrics and step % 10 == 0:
                global_step = step + task_idx * cfg["steps_per_task"]
                metrics.push_step(global_step, task_name, loss_val)

            # Mid-task eval
            if cfg["eval_every"] > 0 and (step + 1) % cfg["eval_every"] == 0:
                model.eval()
                tasks_so_far = tasks[: task_idx + 1]
                mid_results = evaluate_all(model, tokenizer, tasks_so_far, mode=cfg["mode"])
                print_results_table(mid_results)
                model.train()

        # -------------------------------------------------------------------
        # Task boundary: checkpoint + eval
        # -------------------------------------------------------------------

        print(f"\n--- Task {task_idx} ({task_name}) complete ---")

        # Save checkpoint
        save_checkpoint(model, cfg["out_dir"], task_id=task_idx)
        print(f"Checkpoint saved to {cfg['out_dir']}/task_{task_idx}/")

        # Evaluate on all tasks seen so far
        model.eval()
        tasks_so_far = tasks[: task_idx + 1]
        results = evaluate_all(model, tokenizer, tasks_so_far, mode=cfg["mode"])
        results_history.append(results)
        model.train()

        # Print and log
        forgetting = compute_forgetting(results_history)
        print_results_table(results, forgetting=forgetting)
        if metrics:
            bwt_so_far = compute_bwt(results_history)
            metrics.push_eval(results, bwt_so_far)

        # Save results incrementally
        results_path = os.path.join(cfg["out_dir"], "results.json")
        with open(results_path, "w") as f:
            json.dump(
                {"config": cfg, "method": cfg["method"], "history": results_history},
                f,
                indent=2,
                default=str,
            )

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("  FINAL SUMMARY")
    print(f"{'=' * 60}\n")

    forgetting = compute_forgetting(results_history)
    bwt = compute_bwt(results_history)
    retention, plasticity = compute_retention_plasticity(results_history, tasks)
    score = leaderboard_score(retention, plasticity)

    print(f"  Method      : {cfg['method']}")
    print(f"  Memory      : {cfg['memory']}")
    print(f"  Retention   : {retention:.4f}")
    print(f"  Plasticity  : {plasticity:.4f}")
    print(f"  Score        : {score:.4f}")
    print(f"  BWT         : {bwt:.4f}")
    print(f"  Forgetting  : {forgetting}")
    print(f"  Total time  : {time.time() - t0:.1f}s")
    print()

    # Save final results
    final_results = {
        "config": cfg,
        "method": cfg["method"],
        "history": results_history,
        "retention": retention,
        "plasticity": plasticity,
        "score": score,
        "bwt": bwt,
        "forgetting": forgetting,
    }
    results_path = os.path.join(cfg["out_dir"], "results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    # Push final metrics
    if metrics:
        metrics.push_summary(retention, plasticity)
        metrics.close()
