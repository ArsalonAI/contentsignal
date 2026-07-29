# Milestone targets mirror trd.md §16. Every stage is idempotent and skips work when its
# outputs exist with a matching config hash, so re-running a milestone is cheap.
#
# Targets past m2 are placeholders until their CLI commands land; they will fail loudly
# with the milestone that owns them rather than doing nothing.

RUN := uv run
CS  := $(RUN) contentsignal

RET_WINDOWS  := ret_w1 ret_w2 ret_w3 ret_w4
CAND_WINDOWS := rank_w1 rank_w2 rank_w3 rank_w4 val test
ALL_WINDOWS  := $(RET_WINDOWS) $(CAND_WINDOWS)

TABULAR_GROUPS := customer article categorical cross
RANKERS        := lgbm mlp dcn
SEEDS          := 1 2 3

.PHONY: help setup test lint typecheck fmt check clean \
        m0 m1 m2 m3 m4 m5 m6 m7 m8 m9

help:
	@echo "setup      install the 3.11 environment and dev extras"
	@echo "check      lint + typecheck + tests"
	@echo "m0..m9     milestone targets (trd.md §16)"

setup:
	uv sync --extra dev

test:
	$(RUN) pytest

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

fmt:
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

typecheck:
	$(RUN) mypy

check: lint typecheck test

# --- milestones ---------------------------------------------------------------------

# M0: environment + data acquisition. The gate is that the CLI runs.
m0: setup
	$(CS) --help >/dev/null && echo "m0: CLI ok"
	$(CS) splits

# M1: parquet conversion, window roles, leakage tests. Gates everything downstream —
# no model is trained until these pass.
m1: m0
	$(CS) ingest
	$(RUN) pytest tests/test_leakage.py tests/test_splits.py

# M2: cohort, positives, and every tabular feature group across all ten windows.
m2: m1
	$(CS) sample
	$(RUN) pytest tests/test_sampling.py
	@for w in $(ALL_WINDOWS); do \
		for g in $(TABULAR_GROUPS); do \
			$(CS) build-features --group $$g --window $$w || exit 1; \
		done; \
	done

# M3: stage 1. `pop` is the sanity floor and costs nothing — if no trained retriever
# beats it, that is the finding, and it is far cheaper to learn here than at M9.
m3: m2
	$(CS) train-retriever --arm pop
	@for s in $(SEEDS); do \
		$(CS) train-retriever --arm R1 --variant b --seed $$s || exit 1; \
		$(CS) train-retriever --arm R2 --variant b --seed $$s || exit 1; \
	done
	$(CS) train-retriever --arm R2 --variant a --seed 1
	$(CS) train-retriever --arm R2 --variant b --seed 1 --no-logq
	$(CS) embed --retriever R2 --seed 1
	$(RUN) pytest tests/test_retrieval.py

# M4: H2 — the retrieval-level answer, including the whole K sweep from one pass.
m4: m3
	$(CS) evaluate --arm R1 --stage retrieval --split val
	$(CS) evaluate --arm R2 --stage retrieval --split val

# M5: candidates for every window the frozen retriever serves, then retrieval features.
m5: m4
	@for w in $(CAND_WINDOWS); do \
		$(CS) retrieve --window $$w || exit 1; \
		$(CS) build-features --group retrieval --window $$w || exit 1; \
	done

# M6: stage 2 — three rankers on the identical, digest-checked candidate set.
m6: m5
	@for a in $(RANKERS); do \
		for s in $(SEEDS); do \
			$(CS) train-ranker --arm $$a --negatives retrieved --seed $$s || exit 1; \
			$(CS) evaluate --arm $$a --stage e2e --split val || exit 1; \
		done; \
	done

# M7: H3 — the same ranker on random negatives, plus calibration and the business proxy.
m7: m6
	@for s in $(SEEDS); do \
		$(CS) train-ranker --arm dcn --negatives random --seed $$s || exit 1; \
	done
	$(RUN) pytest tests/test_calibration.py tests/test_e2e_metrics.py

# M8: per-stage cost, including the exact-vs-FAISS comparison at a 105k catalog.
m8: m7
	$(CS) bench --config stage1_topk_exact
	$(CS) bench --config stage1_topk_faiss
	$(CS) bench --config e2e_k100
	$(CS) bench --config e2e_k500

# M9 is the only point at which the test split is read.
m9: m8
	$(CS) report

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
