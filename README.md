# PyTorch Architecture Visualizer

Extract and visualize the architecture of a PyTorch model directly from a GitHub repository. This tool analyzes the model's structure using static AST analysis and represents it as an interactive graph.

## 🚀 Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the App**:
   ```bash
   streamlit run sources/app.py
   ```
3. **Analyze**: Enter a GitHub URL in the sidebar and click "Analyze Architecture".

## ✨ Core Features
- **Automatic Model Discovery**: Scans repositories for `nn.Module` subclasses.
- **Static Analysis**: Extracts layers and data flow without executing potentially unsafe code.
- **Interactive Visualization**: Uses a node-editor interface to explore model connectivity.
- **Detailed Insights**: View layer types and parameters in a structured table.

## 📚 Documentation
For more detailed guides, please refer to our documentation:
- [Installation Guide](doc/installation.md)
- [Usage Instructions](doc/usage.md)
- [Building Full Documentation](doc/build_docs.md)

## 🛠️ Project Status
- [x] Basic AST extraction for linear flows.
- [ ] Support for complex conditional branching in `forward`.
- [ ] Enhanced layout engine for automatic node placement.
- [ ] Integration with `torch.fx` for optional dynamic analysis.
