import os
import sys

sys.path.insert(0, os.path.abspath("../../"))
sys.path.insert(0, os.path.abspath("../../sources"))

project = "PyTorch Architecture Visualizer"
copyright = "2026, OpenCode"
author = "OpenCode"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
]

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = []

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autosummary_generate = True
