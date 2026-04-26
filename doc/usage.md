# Usage Instructions

Learn how to use the PyTorch Architecture Visualizer to explore neural network structures.

## 🚀 Launching the Application

To start the app, ensure your virtual environment is activated and run:
```bash
streamlit run sources/app.py
```
The app will open in your default web browser (usually at `http://localhost:8501`).

## 🛠️ How to Analyze a Model

1. **Provide URL**: In the sidebar, paste the URL of a public GitHub repository containing a PyTorch model.
2. **Start Analysis**: Click the **"Analyze Architecture"** button.
3. **Processing**: The app will:
    - Clone the repository to a temporary folder.
    - Search for Python files containing `nn.Module`.
    - Parse the `__init__` method for layers.
    - Parse the `forward` method for connectivity.
4. **Explore the Graph**:
    - **Pan**: Click and drag the background.
    - **Zoom**: Use the mouse wheel to zoom in and out.
    - **Nodes**: Each node represents a layer. The label shows the layer name and its PyTorch type.
5. **View Details**: Open the "View Layer Details" expander at the bottom to see a table of all extracted layers and their parameters.

## 🧪 Example Repositories
Try these types of repositories for best results:
- Repositories with a single, clear `nn.Module` class.
- Models that follow a standard `forward` pass (e.g., Sequential-like structures).

## ⚠️ Limitations
- **Dynamic Flow**: The current version uses static analysis. Models with heavy conditional logic (if/else) or dynamic loop-based connectivity in the `forward` method may not be fully captured.
- **Private Repos**: Only public repositories are supported by default.

[← Back to Home](../README.md)
