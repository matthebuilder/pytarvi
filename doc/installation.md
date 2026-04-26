# Installation Guide

This guide will walk you through setting up the PyTorch Architecture Visualizer on your local machine.

## 💻 System Requirements
- **Python**: 3.8 or higher.
- **Operating System**: Linux, macOS, or Windows.
- **Internet Connection**: Required to clone target GitHub repositories.

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment
We recommend using a virtual environment to avoid dependency conflicts.

**Linux/macOS**:
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows**:
```bash
python -m venv .venv
.venv\\Scripts\\activate
```

## 🛠️ Setup Instructions
... (rest of the content) ...
### 3. Install Dependencies
Install all required packages using pip:
```bash
pip install -r requirements.txt
```

### 4. Verify Installation
You can verify that everything is installed correctly by running the test suite:
```bash
PYTHONPATH=. pytest tests/
```

## 📦 Key Dependencies
- `streamlit`: Powers the web interface.
- `streamlit-node-editor`: Provides the interactive graph visualization.
- `torch`: Used for type references and model understanding.
- `GitPython`: Handles GitHub repository cloning.
- `loguru`: Provides structured logging.

## ❓ Troubleshooting
- **PyTorch Installation**: If you have a GPU, you may want to install the specific version of PyTorch that matches your CUDA version from [pytorch.org](https://pytorch.org/).
- **Git Errors**: Ensure you have `git` installed and configured on your system path.

[← Back to Home](../README.md)
