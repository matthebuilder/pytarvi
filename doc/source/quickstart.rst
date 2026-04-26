Quick Start
===========

1. Start the Streamlit application::

    streamlit run sources/app.py

2. Enter a GitHub repository URL in the sidebar (e.g.
   ``https://github.com/pytorch/vision``).

3. Click **Analyze** to clone the repository and scan for
   ``nn.Module`` subclasses.

4. Select a model class from the dropdown.

5. Choose an input shape preset or enter a custom shape.

6. Click **Extract Architecture** to trace the model and render the
   interactive node graph.

7. Drag nodes to rearrange, right-click to add new nodes, and drag
   between ports to create connections.
