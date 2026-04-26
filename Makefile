.PHONY: run test lint typecheck doc clean

run:
	streamlit run sources/app.py

test:
	pytest tests/ -v

lint:
	ruff check sources/ tests/

typecheck:
	mypy sources/

doc:
	$(MAKE) -C doc html

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .mypy_cache .ruff_cache .pytest_cache
	$(MAKE) -C doc clean
