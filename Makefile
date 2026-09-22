.PHONY: test lint check

test:
	pytest -q

lint:
	ruff check .

check: lint test
