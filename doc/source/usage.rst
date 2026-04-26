Usage Guide
============

The PyTorch Architecture Visualizer provides a simple web interface to explore model structures.

Getting Started
---------------
Run the application using Streamlit:

.. code-block:: bash

   streamlit run sources/app.py

Using the App
-------------
1. **Enter GitHub URL**: Provide the URL of a public repository containing a PyTorch model.
2. **Analyze**: Click "Analyze Architecture". The tool will clone the repo and parse the code.
3. **Interact**: 
   - Use the graph to see layer connectivity.
   - Zoom and pan to navigate large models.
   - Check the "View Layer Details" expander for specific parameters.

Supported Models
----------------
The tool works best with models that inherit from ``nn.Module`` and have a clear ``forward`` method.
