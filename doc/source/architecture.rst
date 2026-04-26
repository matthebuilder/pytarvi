Architecture
============

Data Flow
---------

The application follows this pipeline::

 GitHub URL -> Clone Repo -> Find Model Files -> Import Models
   -> Trace/Extract DAG -> Convert to Node Editor -> Render

Three-Tier Extraction Strategy
------------------------------

The extractor uses three strategies in order of preference:

1. **FX Symbolic Trace** (primary): Uses ``torch.fx.symbolic_trace()``
   to capture all operations including functional calls (``F.relu``,
   ``F.dropout``, etc.) and constructs a complete DAG with edges
   derived from ``node.all_input_nodes``.

2. **Hook-Based Extraction** (fallback): Registers forward hooks on
   every leaf module, runs a dummy forward pass, and collects
   input/output shapes. Functional operations are invisible to hooks.

3. **AST Analysis** (last resort): Parses Python source files, walks
   ``__init__`` for ``self.x = nn.Conv2d(...)`` assignments, and
   builds a linear chain of nodes. No shapes or real edges are
   available.

Module Overview
---------------

``sources/config.py``
    Constants, mappings, and helper functions.

``sources/exceptions.py``
    Custom exception hierarchy.

``sources/graph_model.py``
    Core data model: ``ArchNode``, ``ArchEdge``, ``ArchitectureGraph``.

``sources/layout.py``
    Auto-layout algorithm (topological sort + positioning).

``sources/repo_handler.py``
    GitHub repo cloning, model file discovery, and dynamic import.

``sources/graph_extractor.py``
    Architecture extraction via FX trace, hooks, or AST.

``sources/node_editor_renderer.py``
    Convert ``ArchitectureGraph`` to ``st_node_editor`` arguments.

``sources/app.py``
    Streamlit application entry point.
