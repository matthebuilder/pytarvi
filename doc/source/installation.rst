Installation
============

Prerequisites
-------------

- Python 3.8+
- Git

Setup
-----

1. Clone this repository.

2. Create and activate a virtual environment::

    python3.8 -m venv .venv
    source .venv/bin/activate

3. Install the package with dependencies::

    pip install -e ".[dev,doc]"

4. Verify the installation::

    pytest tests/ -v
