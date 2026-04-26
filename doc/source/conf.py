# PyTorch Architecture Viewer documentation build configuration

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "PyTorch Architecture Viewer"
copyright = "2026, PyTorch Architecture Viewer Contributors"  # noqa: A001
author = "PyTorch Architecture Viewer Contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autoclass_content = "both"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
