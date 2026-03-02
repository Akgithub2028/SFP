"""
scripts/submit_server.py — Leaderboard submission API with SQLite.

Runs on dev-georgios. Accepts method code, benchmarks on Modal, stores in SQLite.

    uvicorn scripts.submit_server:app --host 0.0.0.0 --port 8090

Endpoints:
    POST /submit       — submit a method for benchmarking
    GET  /status/{id}  — poll submission status
    GET  /leaderboard  — current leaderboard (best per method)
    GET  /submissions  — all submissions
"""

from __future__ import annotations

import ast
import hashlib
import json
import sqlite3
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="sfp submissions", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

REPO_DIR = Path.home() / "github" / "paradigmxyz" / "sfp"
DB_PATH = Path.home() / "sfp.db"

BLOCKED_IMPORTS = {
    "os", "subprocess", "sys", "shutil", "pathlib", "socket", "http",
    "urllib", "requests", "importlib", "ctypes", "signal", "multiprocessing",
    "threading", "asyncio", "pickle", "shelve", "tempfile", "glob",
    "webbrowser", "code", "codeop",
}
MAX_CODE_SIZE = 10_000
RATE_LIMIT_SECONDS = 300


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------


def init_db() -> None:
    with get_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS submissions (
                id            TEXT PRIMARY KEY,
                method        TEXT NOT NULL,
                code          TEXT NOT NULL,
                description   TEXT DEFAULT '',
                contributor   TEXT DEFAULT 'anonymous',
                status        TEXT DEFAULT 'queued',
                score_mean    REAL,
                score_std     REAL,
                retention_mean REAL,
                retention_std  REAL,
                plasticity_mean REAL,
                plasticity_std  REAL,
                results_json  TEXT,
                error         TEXT,
                created_at    TEXT NOT NULL,
                completed_at  TEXT
            )
        """)


@contextmanager
def get_db():
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class Submission(BaseModel):
    name: str
    code: str
    description: str = ""
    contributor: str = "anonymous"


class SubmissionResponse(BaseModel):
    id: str
    status: str
    message: str


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_code(name: str, code: str) -> str | None:
    if len(code) > MAX_CODE_SIZE:
        return f"Code too large ({len(code)} bytes, max {MAX_CODE_SIZE})"
    if not name.isidentifier():
        return f"Invalid method name: {name!r}"
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"Syntax error: {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in BLOCKED_IMPORTS:
                    return f"Blocked import: {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module.split(".")[0]
            if mod in BLOCKED_IMPORTS:
                return f"Blocked import: {node.module}"
    func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    loss_fn = f"{name}_loss" if not name.endswith("_loss") else name
    has_loss = any(fn.endswith("_loss") for fn in func_names)
    if name not in func_names and loss_fn not in func_names and not has_loss:
        return f"Code must define a function ending in '_loss' (e.g. '{loss_fn}')"
    return None


# ---------------------------------------------------------------------------
# Benchmark runner (background thread)
# ---------------------------------------------------------------------------


def run_benchmark(sub_id: str, name: str, code: str) -> None:
    try:
        with get_db() as db:
            db.execute("UPDATE submissions SET status='running' WHERE id=?", (sub_id,))

        subprocess.run(["git", "pull", "--ff-only"], cwd=REPO_DIR, capture_output=True)

        result = subprocess.run(
            [
                "modal", "run", "scripts/modal_bench.py",
                "--method", name, "--memory", "128", "--preset", "standard",
                "--code", code,
            ],
            capture_output=True, text=True, cwd=str(REPO_DIR), timeout=5400,
        )

        if result.returncode != 0:
            with get_db() as db:
                db.execute(
                    "UPDATE submissions SET status='failed', error=?, completed_at=? WHERE id=?",
                    (result.stderr[-2000:], _now(), sub_id),
                )
            return

        results_path = REPO_DIR / "benchmark_results.json"
        if not results_path.exists():
            with get_db() as db:
                db.execute(
                    "UPDATE submissions SET status='failed', error='No results file', "
                    "completed_at=? WHERE id=?",
                    (_now(), sub_id),
                )
            return

        bench = json.loads(results_path.read_text())

        with get_db() as db:
            db.execute("""
                UPDATE submissions SET
                    status='complete',
                    score_mean=?, score_std=?,
                    retention_mean=?, retention_std=?,
                    plasticity_mean=?, plasticity_std=?,
                    results_json=?, completed_at=?
                WHERE id=?
            """, (
                bench["score_mean"], bench["score_std"],
                bench["retention_mean"], bench["retention_std"],
                bench["plasticity_mean"], bench["plasticity_std"],
                json.dumps(bench), _now(), sub_id,
            ))

    except Exception as e:
        with get_db() as db:
            db.execute(
                "UPDATE submissions SET status='failed', error=?, completed_at=? WHERE id=?",
                (str(e)[:2000], _now(), sub_id),
            )


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.on_event("startup")
def startup():
    init_db()


@app.post("/submit", response_model=SubmissionResponse)
def submit(sub: Submission):
    err = validate_code(sub.name, sub.code)
    if err:
        raise HTTPException(status_code=400, detail=err)

    # Rate limit by contributor
    with get_db() as db:
        row = db.execute(
            "SELECT created_at FROM submissions WHERE contributor=? "
            "ORDER BY created_at DESC LIMIT 1",
            (sub.contributor,),
        ).fetchone()
        if row:
            last = datetime.fromisoformat(row["created_at"])
            elapsed = (datetime.now(UTC) - last).total_seconds()
            if elapsed < RATE_LIMIT_SECONDS:
                wait = int(RATE_LIMIT_SECONDS - elapsed)
                raise HTTPException(429, f"Rate limited. Try again in {wait}s.")

    sub_id = hashlib.sha256(f"{sub.name}{time.time()}".encode()).hexdigest()[:12]

    with get_db() as db:
        db.execute(
            "INSERT INTO submissions (id, method, code, description, contributor, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (sub_id, sub.name, sub.code, sub.description, sub.contributor, _now()),
        )

    threading.Thread(
        target=run_benchmark, args=(sub_id, sub.name, sub.code), daemon=True,
    ).start()

    return SubmissionResponse(
        id=sub_id, status="queued",
        message=f"Queued. Poll GET /status/{sub_id} for results (~45 min).",
    )


@app.get("/status/{sub_id}")
def status(sub_id: str):
    with get_db() as db:
        row = db.execute("SELECT * FROM submissions WHERE id=?", (sub_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Not found")
    return dict(row)


@app.get("/results/{sub_id}")
def results(sub_id: str):
    """Full benchmark results for a submission (used by dashboard charts)."""
    with get_db() as db:
        row = db.execute(
            "SELECT results_json FROM submissions WHERE id=? AND status='complete'",
            (sub_id,),
        ).fetchone()
    if not row or not row["results_json"]:
        raise HTTPException(404, "No results")
    return json.loads(row["results_json"])


BENCHMARK = {
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "tasks": ["math", "code", "ifeval"],
    "steps_per_task": 1000,
    "lora_rank": 64,
    "seeds": [42, 43, 44],
    "memory": 128,
}


@app.get("/leaderboard")
def leaderboard():
    """Best submission per method, ordered by score."""
    with get_db() as db:
        rows = db.execute("""
            SELECT method, description, contributor, score_mean, score_std,
                   retention_mean, retention_std, plasticity_mean, plasticity_std,
                   completed_at as date
            FROM submissions
            WHERE status='complete'
            AND score_mean = (
                SELECT MAX(s2.score_mean) FROM submissions s2
                WHERE s2.method = submissions.method AND s2.status='complete'
            )
            GROUP BY method
            ORDER BY score_mean DESC
        """).fetchall()
    return {
        "benchmark": BENCHMARK,
        "entries": [dict(r) for r in rows],
    }


@app.get("/leaderboard/details")
def leaderboard_details():
    """Top methods with full per-seed results for charts."""
    with get_db() as db:
        rows = db.execute("""
            SELECT method, contributor, score_mean, results_json
            FROM submissions
            WHERE status='complete'
            AND score_mean = (
                SELECT MAX(s2.score_mean) FROM submissions s2
                WHERE s2.method = submissions.method AND s2.status='complete'
            )
            GROUP BY method
            ORDER BY score_mean DESC
            LIMIT 5
        """).fetchall()
    entries = []
    for r in rows:
        entry = {
            "method": r["method"],
            "contributor": r["contributor"],
            "score_mean": r["score_mean"],
        }
        if r["results_json"]:
            entry["per_seed"] = json.loads(r["results_json"]).get("per_seed", [])
        entries.append(entry)
    return {"entries": entries}


@app.get("/submissions")
def all_submissions(limit: int = 50):
    with get_db() as db:
        rows = db.execute(
            "SELECT id, method, contributor, status, score_mean, created_at, completed_at "
            "FROM submissions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
