PAPER_DIR := paper
LATEX_IMG := sfp-latex
LEAN_DIR := lean-sfp
LEAN_IMG := sfp-lean

.PHONY: paper clean-paper paper-docker-build install lint test fmt all lean lean-build lean-sorry

all: install lint test lean paper

# --- Python ---
install:
	uv sync --all-extras

lint:
	uv run ruff check sfp/ tests/ scripts/ tasks/ *.py

fmt:
	uv run ruff format sfp/ tests/ scripts/ tasks/ *.py
	uv run ruff check --fix sfp/ tests/ scripts/ tasks/ *.py

test:
	uv run pytest

# --- Lean (dockerized) ---
lean-build:
	docker build -t $(LEAN_IMG) $(LEAN_DIR)

lean:
	docker run --rm \
		-v "$(CURDIR)/$(LEAN_DIR)/SFP":/lean-sfp/SFP \
		-v "$(CURDIR)/$(LEAN_DIR)/SFP.lean":/lean-sfp/SFP.lean \
		$(LEAN_IMG) build

lean-sorry:
	@echo "=== sorry count ==="
	@grep -rn sorry $(LEAN_DIR)/SFP/ || echo "No sorry found!"

# --- LaTeX (dockerized) ---
paper-docker-build:
	docker build -t $(LATEX_IMG) $(PAPER_DIR)

paper: paper-docker-build
	docker run --rm -v "$(CURDIR)/$(PAPER_DIR)":/paper $(LATEX_IMG)
	cp $(PAPER_DIR)/build/main.pdf sfp.pdf
	cp $(PAPER_DIR)/build/main.pdf $(PAPER_DIR)/sfp.pdf
	@echo "\n→ sfp.pdf and $(PAPER_DIR)/sfp.pdf"

clean-paper:
	docker run --rm -v "$(CURDIR)/$(PAPER_DIR)":/paper --entrypoint rm $(LATEX_IMG) -rf /paper/build 2>/dev/null || rm -rf $(PAPER_DIR)/build
	rm -f sfp.pdf $(PAPER_DIR)/sfp.pdf
