"""Tests for sources.graph_model — dataclass construction and queries."""


import pytest

from sources.graph_model import ArchEdge, ArchitectureGraph, ArchNode


class TestArchNode:
    """Tests for ArchNode frozen dataclass."""

    def test_construction(self) -> None:
        node = ArchNode(
            id="conv1", name="conv1", op_type="Conv2d",
            category="Convolution", color="#3b82f6",
        )
        assert node.id == "conv1"
        assert node.op_type == "Conv2d"
        assert node.input_shape is None
        assert node.output_shape is None
        assert node.params == {}

    def test_with_shapes(self) -> None:
        node = ArchNode(
            id="fc1", name="fc1", op_type="Linear", category="Linear",
            input_shape=(1, 16), output_shape=(1, 10), color="#60a5fa",
        )
        assert node.input_shape == (1, 16)
        assert node.output_shape == (1, 10)

    def test_with_params(self) -> None:
        node = ArchNode(
            id="drop1", name="drop1", op_type="Dropout", category="Dropout",
            params={"p": 0.5}, color="#fb923c",
        )
        assert node.params["p"] == 0.5

    def test_frozen(self) -> None:
        node = ArchNode(
            id="x", name="x", op_type="ReLU",
            category="Activation", color="#4ade80",
        )
        with pytest.raises(AttributeError):
            node.id = "y"  # type: ignore[misc]

    def test_default_optional_fields(self) -> None:
        node = ArchNode(
            id="n", name="n", op_type="Conv2d",
            category="Convolution", color="#000",
        )
        assert node.input_shape is None
        assert node.output_shape is None
        assert node.params == {}


class TestArchEdge:
    """Tests for ArchEdge frozen dataclass."""

    def test_construction(self) -> None:
        edge = ArchEdge(id="e0", source_id="a", target_id="b")
        assert edge.source_id == "a"
        assert edge.target_id == "b"
        assert edge.source_port == 0
        assert edge.target_port == 0

    def test_custom_ports(self) -> None:
        edge = ArchEdge(
            id="e1", source_id="a", source_port=1,
            target_id="b", target_port=2,
        )
        assert edge.source_port == 1
        assert edge.target_port == 2

    def test_frozen(self) -> None:
        edge = ArchEdge(id="e0", source_id="a", target_id="b")
        with pytest.raises(AttributeError):
            edge.source_id = "c"  # type: ignore[misc]


class TestArchitectureGraph:
    """Tests for ArchitectureGraph."""

    def test_empty_graph(self) -> None:
        graph = ArchitectureGraph()
        assert graph.nodes == []
        assert graph.edges == []
        assert graph.model_name == ""

    def test_node_by_id(self) -> None:
        node = ArchNode(
            id="conv1", name="conv1", op_type="Conv2d",
            category="Convolution", color="#3b82f6",
        )
        graph = ArchitectureGraph(nodes=[node])
        assert graph.node_by_id("conv1") is node
        assert graph.node_by_id("missing") is None

    def test_edges_from(self, sample_arch_graph: ArchitectureGraph) -> None:
        edges = sample_arch_graph.edges_from("conv1")
        assert len(edges) == 1
        assert edges[0].target_id == "relu1"

    def test_edges_to(self, sample_arch_graph: ArchitectureGraph) -> None:
        edges = sample_arch_graph.edges_to("fc1")
        assert len(edges) == 1
        assert edges[0].source_id == "relu1"

    def test_unique_op_types(self, sample_arch_graph: ArchitectureGraph) -> None:
        op_types = sample_arch_graph.unique_op_types()
        assert op_types == ["Conv2d", "Linear", "ReLU"]

    def test_edges_from_empty(self, sample_arch_graph: ArchitectureGraph) -> None:
        assert sample_arch_graph.edges_from("fc1") == []

    def test_edges_to_empty(self, sample_arch_graph: ArchitectureGraph) -> None:
        assert sample_arch_graph.edges_to("conv1") == []
