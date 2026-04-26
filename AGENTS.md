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

## Coding Rules
- MUST follow PEP 8 style guidelines
- Use `@dataclass` where possible
- Keep classes focused on a single responsibility
- Prefer composition over inheritance
- MUST follow PEP 484 for type hints where possible
- MUST use meaningful, descriptive variable and function names
- MUST use 4 spaces for indentation (never tabs)
- NEVER use emoji, or unicode that emulates emoji (e.g. ✓, ✗). The only exception is when testing the impact of multibyte characters.

### Documentation
- MUST avoid including redundant comments which are tautological or self-demonstating (e.g. cases where it is easily parsable what the code does at a glance so the comment does)
- MUST include docstrings for all public functions, classes, and methods
- MUST document function parameters, return values, and exceptions raised
- Keep comments up-to-date with code changes
- Include examples in docstrings for complex functions

### Best Practice
- NEVER use mutable default arguments
- Use context managers for file/resource management
- MUST use is for comparing with None, True, False
- MUST use f-strings for string formatting
- Use list comprehensions and generator expressions where possible
￼ 
## Security
- NEVER store secrets, API keys, or passwords in code. Only store them in .env
Ensure .env is declared in .gitignore.
- MUST use environment variables for sensitive configuration
- NEVER log sensitive information (passwords, tokens)

## Before Committing
- [ ] All tests run and pass
- [ ] Type checking passes (mypy)
- [ ] All functions have docstrings and type hints
- [ ] No commented-out code or debug statements

Remember: Prioritize clarity and maintainability over cleverness. This is your core directive.
