import pytest
from sources.mapper import GraphMapper
from sources.models import Node, Edge


class TestGraphMapper:
    def test_map_to_editor_format(self):
        mapper = GraphMapper()
        nodes = [
            Node(id="l1", name="conv1", type="Conv2d"),
            Node(id="l2", name="relu1", type="ReLU"),
        ]
        edges = [
            Edge(id="e1", source="l1", target="l2"),
        ]

        node_defs, initial_nodes, initial_connections = mapper.map_to_editor(
            nodes, edges
        )

        assert "Conv2d" in node_defs
        assert "ReLU" in node_defs
        assert len(initial_nodes) == 2
        assert len(initial_connections) == 1

        # Check node structure
        node_0 = initial_nodes[0]
        assert node_0["id"] == "l1"
        assert node_0["type"] == "Conv2d"
        assert "x" in node_0
        assert "y" in node_0

        # Check edge structure
        edge_0 = initial_connections[0]
        assert edge_0["id"] == "e1"
        assert edge_0["fromNode"] == "l1"
        assert edge_0["toNode"] == "l2"

    def test_orphan_nodes(self):
        mapper = GraphMapper()
        nodes = [
            Node(id="l1", name="conv1", type="Conv2d"),
            Node(id="l2", name="relu1", type="ReLU"),
        ]
        edges = []  # No edges

        node_defs, initial_nodes, initial_connections = mapper.map_to_editor(
            nodes, edges
        )
        assert len(initial_nodes) == 2
        assert len(initial_connections) == 0

    def test_invalid_edges(self):
        mapper = GraphMapper()
        nodes = [
            Node(id="l1", name="conv1", type="Conv2d"),
        ]
        edges = [
            Edge(id="e1", source="l1", target="nonexistent"),
        ]

        node_defs, initial_nodes, initial_connections = mapper.map_to_editor(
            nodes, edges
        )
        # Edge should be filtered out if target doesn't exist
        assert len(initial_connections) == 0
