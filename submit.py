"""
submit.py — Submit to the sfp leaderboard. No PR, no GPU needed.

Usage:
    python submit.py submissions/my_method.py
    python submit.py submissions/my_method.py --no-wait

Your file should look like:

    '''One-line description.'''
    CONTRIBUTOR = "Your Name"

    def my_method_loss(model, batch, **kw):
        ...
        return loss

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
from pathlib import Path

SERVER = "http://sfp.paradigm.xyz:8090"


def parse_submission(path: str) -> tuple[str, str, str, str]:
    """Parse a submission file. Returns (name, code, description, contributor)."""
    p = Path(path)
    if not p.exists():
        print(f"Error: {path} not found")
        sys.exit(1)

    code = p.read_text()

    # Syntax check
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        print(f"Syntax error in {path}: {e}")
        sys.exit(1)

    # Find the loss function (must end with _loss)
    loss_fns = [
        n.name for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name.endswith("_loss")
    ]
    if not loss_fns:
        print(f"Error: {path} must define a function ending in '_loss'")
        print("Example: def my_method_loss(model, batch, **kw): ...")
        sys.exit(1)
    if len(loss_fns) > 1:
        print(f"Warning: multiple loss functions found, using {loss_fns[0]}")

    fn_name = loss_fns[0]
    name = fn_name.removesuffix("_loss")

    # Extract CONTRIBUTOR
    contributor = "anonymous"
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (isinstance(target, ast.Name) and target.id == "CONTRIBUTOR"
                        and isinstance(node.value, ast.Constant)):
                    contributor = str(node.value.value)

    # Extract description from module docstring
    description = ""
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)):
        description = str(tree.body[0].value.value).strip()

    return name, code, description, contributor


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
    parser = argparse.ArgumentParser(
        description="Submit to the sfp leaderboard",
        usage="python submit.py submissions/my_method.py",
    )
    parser.add_argument("file", help="Path to your submission file")
    parser.add_argument("--server", default=SERVER)
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    global SERVER
    SERVER = args.server

    # Parse
    name, code, description, contributor = parse_submission(args.file)
    print(f"  Method      : {name}")
    print(f"  Contributor : {contributor}")
    print(f"  Description : {description or '(none)'}")
    print(f"  Code size   : {len(code)} bytes")
    print()

    # Submit
    print(f"Submitting to {SERVER}...")
    try:
        resp = post("/submit", {
            "name": name,
            "code": code,
            "description": description,
            "contributor": contributor,
        })
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"Error {e.code}: {body}")
        sys.exit(1)

    sub_id = resp["id"]
    print(f"  Queued: {sub_id}")
    print(f"  Status: {SERVER}/status/{sub_id}\n")

    if args.no_wait:
        print(f"Poll later: curl {SERVER}/status/{sub_id}")
        return

    # Poll
    print("Waiting for benchmark (~45 min)...")
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
            sc = f"{s.get('score_mean', 0):.4f} +/- {s.get('score_std', 0):.4f}"
            rt = f"{s.get('retention_mean', 0):.4f} +/- {s.get('retention_std', 0):.4f}"
            pl = f"{s.get('plasticity_mean', 0):.4f} +/- {s.get('plasticity_std', 0):.4f}"
            print(f"\n\n{'=' * 50}")
            print("  RESULTS")
            print(f"{'=' * 50}")
            print(f"  Method     : {name}")
            print(f"  Score      : {sc}")
            print(f"  Retention  : {rt}")
            print(f"  Plasticity : {pl}")
            print(f"{'=' * 50}")
            print("\n  Your score is on the leaderboard!")
            return

        if status == "failed":
            print(f"\n\nFailed: {s.get('error', 'unknown')}")
            sys.exit(1)


if __name__ == "__main__":
    main()
