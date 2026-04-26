from typing import List, Dict, Any, Tuple
from sources.models import Node, Edge


class GraphMapper:
    """Maps extracted architecture nodes and edges to streamlit-node-editor format."""

    def map_to_editor(
        self, nodes: List[Node], edges: List[Edge]
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Transforms internal Node and Edge lists into the format required by streamlit-node-editor.

        Returns:
            A tuple containing (node_defs, initial_nodes, initial_connections).
        """
        node_defs = {}
        initial_nodes = []
        initial_connections = []

        # 1. Create node definitions based on extracted types
        for node in nodes:
            if node.type not in node_defs:
                node_defs[node.type] = {
                    "category": "PyTorch Layer",
                    "headerColor": "#38bdf8",
                    "inputs": [{"name": "input", "type": "TENSOR"}],
                    "outputs": [{"name": "output", "type": "TENSOR"}],
                    "params": [],  # We could map node.params here, but st_node_editor expects a registry
                }

        # 2. Create initial nodes
        for i, node in enumerate(nodes):
            initial_nodes.append(
                {
                    "id": node.id,
                    "type": node.type,
                    "x": 250.0,
                    "y": i * 150.0,
                    "params": node.params,
                }
            )

        # 3. Create initial connections
        for edge in edges:
            # Only add edge if both source and target exist in the nodes list
            if edge.source in [n.id for n in nodes] and edge.target in [
                n.id for n in nodes
            ]:
                initial_connections.append(
                    {
                        "id": edge.id,
                        "fromNode": edge.source,
                        "fromPort": 0,  # Default to first output port
                        "toNode": edge.target,
                        "toPort": 0,  # Default to first input port
                    }
                )

        return node_defs, initial_nodes, initial_connections
