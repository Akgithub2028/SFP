# sfp

ML paper monorepo — LaTeX + Python experiments.

## Setup

```bash
uv sync --all-extras    # install all deps (Python 3.12)
```

## Project structure

```
sfp/              # Python package (models, data, training, eval)
tests/            # pytest tests
scripts/          # standalone experiment scripts
paper/            # LaTeX source
  main.tex        # entry point
  sections/       # one .tex per section
  figures/        # static + PaperBanana-generated figures
  references.bib  # bibliography
```

## Commands

| Command             | What it does                              |
|---------------------|-------------------------------------------|
| `make install`      | `uv sync --all-extras`                    |
| `make lint`         | ruff check                                |
| `make fmt`          | ruff format + autofix                     |
| `make test`         | pytest                                    |
| `make paper`        | compile LaTeX → `paper/main.pdf` (Docker) |
| `make clean-paper`  | remove LaTeX build artifacts              |

The paper build is fully dockerized — no local TeX install needed, just Docker.

## Figures with PaperBanana

Configure `paper/paperbanana.yaml` with your API key, then use PaperBanana
to generate methodology diagrams and plots from text descriptions.
