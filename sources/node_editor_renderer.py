"""Convert ArchitectureGraph to streamlit-node-editor arguments and render."""

from typing import Any, Dict, List, Optional, Tuple

from streamlit_node_editor import st_node_editor

from sources.config import DEFAULT_HEIGHT, PORT_TYPE, get_category, get_color
from sources.graph_model import ArchitectureGraph, ArchNode
from sources.layout import compute_layout


def build_node_defs(graph: ArchitectureGraph) -> Dict[str, Dict[str, Any]]:
    """Generate node_defs for the streamlit node editor.

    Creates one node definition per unique operation type in the graph.
    Each definition includes typed input/output ports and display
    parameters for layer dimensions.

    Args:
        graph: The :class:`ArchitectureGraph` to render.

    Returns:
        Dictionary mapping operation type names to node definition
        dicts compatible with :func:`st_node_editor`.
    """
    node_defs: Dict[str, Dict[str, Any]] = {}

    for op_type in graph.unique_op_types():
        category = get_category(op_type)
        color = get_color(category)

        sample_node = _find_sample_node(graph, op_type)
        param_defs = _build_param_defs(sample_node)

        node_defs[op_type] = {
            "category": category,
            "headerColor": color,
            "inputs": [{"name": "input", "type": PORT_TYPE}],
            "outputs": [{"name": "output", "type": PORT_TYPE}],
            "params": param_defs,
        }

    return node_defs


def build_initial_nodes(
    graph: ArchitectureGraph,
    layout: Optional[Dict[str, Tuple[float, float]]] = None,
) -> List[Dict[str, Any]]:
    """Generate initial_nodes for the streamlit node editor.

    Maps each :class:`ArchNode` to a positioned node dict with
    computed ``(x, y)`` coordinates from the layout algorithm.

    Args:
        graph: The :class:`ArchitectureGraph` to render.
        layout: Optional pre-computed layout mapping. If ``None``,
            :func:`~sources.layout.compute_layout` is called.

    Returns:
        List of node dicts compatible with :func:`st_node_editor`.
    """
    if layout is None:
        layout = compute_layout(graph.nodes, graph.edges)

    initial_nodes: List[Dict[str, Any]] = []
    for node in graph.nodes:
        pos = layout.get(node.id, (0.0, 0.0))
        params = _node_params_for_display(node)

        initial_nodes.append({
            "id": node.id,
            "type": node.op_type,
            "x": pos[0],
            "y": pos[1],
            "params": params,
        })

    return initial_nodes


def build_initial_connections(graph: ArchitectureGraph) -> List[Dict[str, Any]]:
    """Generate initial_connections for the streamlit node editor.

    Maps each :class:`ArchEdge` to a connection dict referencing
    node IDs and port indices.

    Args:
        graph: The :class:`ArchitectureGraph` to render.

    Returns:
        List of connection dicts compatible with :func:`st_node_editor`.
    """
    connections: List[Dict[str, Any]] = []
    for edge in graph.edges:
        connections.append({
            "id": edge.id,
            "fromNode": edge.source_id,
            "fromPort": edge.source_port,
            "toNode": edge.target_id,
            "toPort": edge.target_port,
        })

    return connections


def render_graph(
    graph: ArchitectureGraph,
    height: int = DEFAULT_HEIGHT,
    key: str = "arch_graph",
) -> Optional[Dict[str, Any]]:
    """Render an architecture graph as an interactive node editor.

    Combines :func:`build_node_defs`, :func:`build_initial_nodes`, and
    :func:`build_initial_connections` into a single call to
    :func:`st_node_editor`.

    Args:
        graph: The :class:`ArchitectureGraph` to render.
        height: Canvas height in pixels.
        key: Unique Streamlit widget key.

    Returns:
        The graph state dict from the node editor, or ``None`` before
        any user interaction.
    """
    node_defs = build_node_defs(graph)
    initial_nodes = build_initial_nodes(graph)
    initial_connections = build_initial_connections(graph)

    return st_node_editor(  # type: ignore[no-any-return]
        node_defs,
        initial_nodes=initial_nodes,
        initial_connections=initial_connections,
        height=height,
        key=key,
    )


def _find_sample_node(graph: ArchitectureGraph, op_type: str) -> Optional[ArchNode]:
    """Find a representative node for a given operation type.

    Args:
        graph: The architecture graph to search.
        op_type: The operation type to find.

    Returns:
        The first :class:`ArchNode` matching the type, or ``None``.
    """
    for node in graph.nodes:
        if node.op_type == op_type:
            return node
    return None


def _build_param_defs(node: Optional[ArchNode]) -> List[Dict[str, Any]]:
    """Build parameter definitions for a node type from a sample node.

    Args:
        node: A sample :class:`ArchNode` to extract param keys from,
            or ``None`` if no sample is available.

    Returns:
        List of parameter definition dicts for the node editor.
    """
    param_defs: List[Dict[str, Any]] = []

    if node is None:
        return param_defs

    for key, value in node.params.items():
        if isinstance(value, bool):
            param_defs.append({
                "key": key,
                "label": _format_label(key),
                "type": "string",
                "default": str(value),
            })
        elif isinstance(value, int):
            param_defs.append({
                "key": key,
                "label": _format_label(key),
                "type": "int",
                "default": value,
            })
        elif isinstance(value, float):
            param_defs.append({
                "key": key,
                "label": _format_label(key),
                "type": "float",
                "default": value,
            })
        elif isinstance(value, (tuple, list)):
            param_defs.append({
                "key": key,
                "label": _format_label(key),
                "type": "string",
                "default": str(list(value)),
            })
        else:
            param_defs.append({
                "key": key,
                "label": _format_label(key),
                "type": "string",
                "default": str(value),
            })

    if node.input_shape is not None:
        param_defs.append({
            "key": "_input_shape",
            "label": "Input Shape",
            "type": "string",
            "default": str(list(node.input_shape)),
        })

    if node.output_shape is not None:
        param_defs.append({
            "key": "_output_shape",
            "label": "Output Shape",
            "type": "string",
            "default": str(list(node.output_shape)),
        })

    return param_defs


def _node_params_for_display(node: ArchNode) -> Dict[str, Any]:
    """Build the params dict for a specific node instance.

    Combines layer parameters with shape information for inline display.

    Args:
        node: The :class:`ArchNode` to build params for.

    Returns:
        Dictionary of parameter key-value pairs for the node.
    """
    params: Dict[str, Any] = {}

    for key, value in node.params.items():
        if isinstance(value, (tuple, list)):
            params[key] = str(list(value))
        else:
            params[key] = value

    if node.input_shape is not None:
        params["_input_shape"] = str(list(node.input_shape))

    if node.output_shape is not None:
        params["_output_shape"] = str(list(node.output_shape))

    return params


def _format_label(key: str) -> str:
    """Format a snake_case parameter key into a display label.

    Args:
        key: The parameter key (e.g. ``"out_channels"``).

    Returns:
        Title-cased label (e.g. ``"Out Channels"``).
    """
    return key.replace("_", " ").title()
