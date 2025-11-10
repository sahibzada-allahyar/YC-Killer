.PHONY: setup run test lean-check lint

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
	python -m grantlab run examples/simple_experiment/config.yml

lean-check:
	python -m grantlab lean-check examples/simple_experiment/config.yml

test:
	pytest -q
