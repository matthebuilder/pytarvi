Technical Architecture
========================

The application follows a linear pipeline to transform raw code into a visual graph.

Pipeline Flow
-------------
1. **GitHub Manager**: Clones the repository and searches for a Python file containing ``nn.Module``.
2. **Architecture Extractor**: 
   - Uses the ``ast`` module for static analysis.
   - Parses ``__init__`` to discover layers.
   - Parses ``forward`` to determine the data flow between layers.
3. **Graph Mapper**: Converts the internal node/edge representation into the format required by the ``streamlit-node-editor``.
4. **Streamlit UI**: Renders the final interactive graph.

Design Decisions
----------------
- **Static Analysis**: Chosen over dynamic analysis (like ``torch.fx``) to ensure security, as the app does not execute untrusted code from GitHub.
- **Linear Layout**: Initially implements a vertical linear layout for simplicity and readability.
