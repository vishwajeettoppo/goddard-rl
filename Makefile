.PHONY: setup validate baselines train eval plots site all clean-checkpoints

PY := .venv/bin/python
UPDATES ?= 120
PARALLEL ?= 3
SEEDS ?= 42,101,2026

setup:
	python -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

validate:
	$(PY) scripts/validate_env.py

baselines:
	$(PY) scripts/run_baselines.py

train:
	SEEDS=$(SEEDS) bash scripts/run_all_training.sh $(UPDATES) $(PARALLEL)

eval:
	$(PY) src/evaluate.py --ablations all

plots:
	$(PY) src/plots.py

site:
	$(PY) scripts/build_site.py

all: validate baselines train eval plots site

lint:
	.venv/bin/ruff check src scripts

clean-checkpoints:
	rm -f results/checkpoints/*.pt results/checkpoints/*.jsonl results/checkpoints/*.meta.json
