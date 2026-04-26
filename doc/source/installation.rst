Installation
============

This guide describes how to set up the environment for the PyTorch Architecture Visualizer.

Prerequisites
-------------
- Python 3.8+
- Git installed on your system

Setup Steps
-----------
1. Clone the repository:
   .. code-block:: bash

      git clone <repo-url>
      cd <repo-folder>

2. Create a virtual environment:
   .. code-block:: bash

      python -m venv .venv
      source .venv/bin/activate  # Linux/macOS
      # .venv\\Scripts\\activate # Windows

    3. Install dependencies:
       .. code-block:: bash

          pip install -r requirements.txt

    4. Verify Installation:
       .. code-block:: bash

          PYTHONPATH=. pytest tests/

