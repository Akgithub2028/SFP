PAPER_DIR := paper
DOCKER_IMG := sfp-latex

.PHONY: paper clean-paper paper-docker paper-docker-build install lint test fmt all

all: install lint test paper

# --- Python ---
install:
	uv sync --all-extras

lint:
	uv run ruff check sfp/ tests/ scripts/

fmt:
	uv run ruff format sfp/ tests/ scripts/
	uv run ruff check --fix sfp/ tests/ scripts/

test:
	uv run pytest

# --- LaTeX (dockerized) ---
paper-docker-build:
	docker build -t $(DOCKER_IMG) $(PAPER_DIR)

paper: paper-docker-build
	docker run --rm -v $(CURDIR)/$(PAPER_DIR):/paper $(DOCKER_IMG)
	cp $(PAPER_DIR)/build/main.pdf sfp.pdf
	@echo "\n→ sfp.pdf"

clean-paper:
	rm -rf $(PAPER_DIR)/build sfp.pdf
