# Project Instructions

This is a streamlit app that uses a user-given github repository's code to extract the pytorch architecture encoded by that code and represent it as a graph using (interactive) nodes.

## Tech stack
- python3.8+
- pypi module `streamlit-node-editor` (github: https://github.com/RhythrosaLabs/streamlit-node-editor)
- logging with loguru
- documentation via Sphinx (google style docstrings)
- use a virtual environment to install modules and run code 

## Design and Planning Guidelines
Take inspiration from how other projects on github with permissive license extract the state graph from pytorch code, but plan implementing without additional modules import, only if absolutely necessary (or the modules are built-in or very established modules, e.g. numpy). The graph representation should include functional elements of the neural network and training (e.g. normalisation, dropout, relu, encoding, layers with type and dimension) and follow the data from input to output.

## Project Architecture
- code lives in `./sources/`
- folder `./doc/` is for documentation
- tests go in `./tests/`
- use `./.venv/` for the local project python environment 

## Development Workflow

### Environment Setup
- Create venv: `python -m venv .venv`
- Activate venv: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows)
- Install dependencies: `pip install -r requirements.txt`

### Quality Assurance Commands
- Linting: `ruff check .`
- Formatting: `ruff format .`
- Type Checking: `mypy sources/`
- Run all tests: `pytest`
- Run a single test file: `pytest tests/test_module.py`
- Run a specific test function: `pytest tests/test_module.py::test_function_name`

### Documentation
- Build HTML docs: `sphinx-build -b html doc/source doc/build`
- Clean build: `rm -rf doc/build`

## Coding Rules

### General Guidelines
- MUST follow PEP 8 style guidelines
- Use `@dataclass` where possible
- Keep classes focused on a single responsibility
- Prefer composition over inheritance
- MUST follow PEP 484 for type hints where possible
- MUST use meaningful, descriptive variable and function names
- MUST use 4 spaces for indentation (never tabs)
- NEVER use emoji, or unicode that emulates emoji (e.g. ✓, ✗). The only exception is when testing the impact of multibyte characters.

### Formatting & Style
- Imports: 
    - MUST be grouped in the following order: Standard library, Third-party libraries, Local application imports.
    - MUST be sorted alphabetically within each group.
- String Formatting: MUST use f-strings for string formatting.
- Line Length: Prefer 88 characters (Black/Ruff standard).

### Type Hinting
- MUST use explicit type hints for all function signatures.
- Use `Optional[T]` for values that can be `None`.
- Use `Union[T, U]` or `T | U` (Python 3.10+) for multiple possible types.
- Avoid `Any` unless absolutely necessary; use `Protocol` or generic types if possible.

### Naming Conventions
- Variables & Functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: Prefix with a single underscore `_private_member`
- Protected members: Prefix with a single underscore (same as private in Python)

### Error Handling
- Use specific exceptions; NEVER use a bare `except:` or `except Exception:`.
- Define custom exception classes for domain-specific errors.
- Use `loguru` for all logging; avoid using the standard `logging` module.
- Log errors with sufficient context but NEVER log sensitive data.
- Use `try...except...finally` or context managers to ensure resource cleanup.

### Documentation
- MUST avoid including redundant comments which are tautological or self-demonstrating.
- MUST include docstrings for all public functions, classes, and methods.
- Use Google style docstrings for all documentation.
- MUST document function parameters, return values, and exceptions raised.
- Keep comments up-to-date with code changes.
- Include examples in docstrings for complex functions.

### Best Practice
- NEVER use mutable default arguments (e.g., use `None` and initialize inside the function).
- Use context managers (`with` statement) for file/resource management.
- MUST use `is` for comparing with `None`, `True`, `False`.
- Use list comprehensions and generator expressions where possible for conciseness.

## Security
- NEVER store secrets, API keys, or passwords in code. Only store them in `.env`.
- Ensure `.env` is declared in `.gitignore`.
- MUST use environment variables for sensitive configuration.
- NEVER log sensitive information (passwords, tokens).

## Before Committing
- [ ] All tests run and pass: `pytest`
- [ ] Linting passes: `ruff check .`
- [ ] Formatting is correct: `ruff format .`
- [ ] Type checking passes: `mypy sources/`
- [ ] All functions have docstrings and type hints
- [ ] No commented-out code or debug statements

Remember: Prioritize clarity and maintainability over cleverness. This is your core directive.
