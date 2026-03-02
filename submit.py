"""
submit.py — Submit your method to the sfp leaderboard. No PR needed.

Usage:
    1. Add your method to methods.py
    2. python submit.py --name my_method --contributor "Your Name"

The benchmark runs on a GPU in the cloud (~45 min). Your score appears
on the leaderboard automatically.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import urllib.error
import urllib.request

SERVER = "http://sfp.paradigm.xyz:8090"


def extract_method(name: str, path: str = "methods.py") -> str:
    """Extract function source + needed imports from methods.py."""
    with open(path) as f:
        source = f.read()

    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)

    loss_fn = f"{name}_loss" if not name.endswith("_loss") else name
    targets = {name, loss_fn}

    # Collect imports
    imports = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            end = node.end_lineno if hasattr(node, "end_lineno") else node.lineno
            imports.append("".join(lines[node.lineno - 1 : end]))

    # Collect function(s)
    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in targets:
            start = node.lineno - 1
            end = node.end_lineno if hasattr(node, "end_lineno") else len(lines)
            funcs.append("".join(lines[start:end]))

    if not funcs:
        all_fns = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        print(f"Error: no function '{name}' or '{loss_fn}' in {path}")
        print(f"Found: {', '.join(all_fns)}")
        sys.exit(1)

    parts = []
    if imports:
        parts.append("".join(imports))
    parts.extend(funcs)
    return "\n\n".join(parts)


def post(endpoint: str, data: dict) -> dict:
    req = urllib.request.Request(
        f"{SERVER}{endpoint}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def get(endpoint: str) -> dict:
    with urllib.request.urlopen(f"{SERVER}{endpoint}") as resp:
        return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Submit to the sfp leaderboard")
    parser.add_argument("--name", required=True, help="Method name (as registered in METHODS)")
    parser.add_argument("--contributor", default="anonymous", help="Your name / handle")
    parser.add_argument("--description", default="", help="One-line description")
    parser.add_argument("--file", default="methods.py", help="File containing your method")
    parser.add_argument("--server", default=SERVER, help="Submission server URL")
    parser.add_argument("--no-wait", action="store_true", help="Submit and exit without waiting")
    args = parser.parse_args()

    global SERVER
    SERVER = args.server

    # Extract code
    print(f"Extracting '{args.name}' from {args.file}...")
    code = extract_method(args.name, args.file)
    print(f"  {len(code)} bytes, {code.count(chr(10))} lines\n")

    # Submit
    print(f"Submitting to {SERVER}...")
    try:
        resp = post("/submit", {
            "name": args.name,
            "code": code,
            "description": args.description,
            "contributor": args.contributor,
        })
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"Error {e.code}: {body}")
        sys.exit(1)

    sub_id = resp["id"]
    print(f"  Queued: {sub_id}")
    print(f"  Status: {SERVER}/status/{sub_id}\n")

    if args.no_wait:
        print("Use --no-wait was set. Check status later:")
        print(f"  curl {SERVER}/status/{sub_id}")
        return

    # Poll for results
    print("Waiting for benchmark (~45 min for standard preset)...")
    spinner = "|/-\\"
    i = 0
    while True:
        time.sleep(10)
        try:
            s = get(f"/status/{sub_id}")
        except Exception:
            i += 1
            continue

        status = s.get("status", "unknown")
        sys.stdout.write(f"\r  {spinner[i % 4]} {status}  ")
        sys.stdout.flush()
        i += 1

        if status == "complete":
            print("\n")
            print("=" * 50)
            print("  RESULTS")
            print("=" * 50)
            print(f"  Method     : {args.name}")
            sc = f"{s.get('score_mean', 0):.4f} +/- {s.get('score_std', 0):.4f}"
            rt = f"{s.get('retention_mean', 0):.4f} +/- {s.get('retention_std', 0):.4f}"
            pl = f"{s.get('plasticity_mean', 0):.4f} +/- {s.get('plasticity_std', 0):.4f}"
            print(f"  Score      : {sc}")
            print(f"  Retention  : {rt}")
            print(f"  Plasticity : {pl}")
            print("=" * 50)
            print("\n  Your score is now on the leaderboard!")
            return

        if status == "failed":
            print(f"\n\nBenchmark failed: {s.get('error', 'unknown error')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
