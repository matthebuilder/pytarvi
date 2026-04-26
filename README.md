# PyTorch Architecture Visualizer (pytarvi)

This repository contains several code experiments of the same project, each implemented using different LLM agent harnesses to evaluate their software engineering capabilities.

## Project Overview
`pytarvi` is a Streamlit application that extracts PyTorch model architectures from GitHub repositories and represents them as interactive node graphs.

## Experiments
- **`pytarvi-glm51`**: Implements a 3-tier extraction strategy (FX trace, hooks, and AST) with auto-layout and comprehensive Sphinx documentation.
- **`pytarvi-gemma4`**: Focuses on static AST analysis for linear flows, automatic model discovery, and an interactive node-editor interface.

Each branch preserves the full commit history of the respective experiment.
