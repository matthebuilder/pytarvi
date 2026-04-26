"""Tests for sources.node_editor_renderer — node_defs, initial_nodes, connections."""



from sources.graph_model import ArchEdge, ArchitectureGraph, ArchNode
from sources.node_editor_renderer import (
    build_initial_connections,
    build_initial_nodes,
    build_node_defs,
)


def _make_graph() -> ArchitectureGraph:
    """Create a minimal ArchitectureGraph for testing."""
    nodes = [
        ArchNode(
            id="conv1", name="conv1", op_type="Conv2d", category="Convolution",
            input_shape=(1, 3, 32, 32), output_shape=(1, 16, 32, 32),
            params={"in_channels": 3, "out_channels": 16, "kernel_size": (3, 3)},
            color="#3b82f6",
        ),
        ArchNode(
            id="relu1", name="relu1", op_type="ReLU", category="Activation",
            input_shape=(1, 16, 32, 32), output_shape=(1, 16, 32, 32),
            color="#4ade80",
        ),
        ArchNode(
            id="fc1", name="fc1", op_type="Linear", category="Linear",
            input_shape=(1, 16), output_shape=(1, 10),
            params={"in_features": 16, "out_features": 10},
            color="#60a5fa",
        ),
    ]
    edges = [
        ArchEdge(
            id="e0", source_id="conv1", source_port=0,
            target_id="relu1", target_port=0,
        ),
        ArchEdge(
            id="e1", source_id="relu1", source_port=0,
            target_id="fc1", target_port=0,
        ),
    ]
    return ArchitectureGraph(
        nodes=nodes, edges=edges, model_name="TestModel",
        source_repo="https://github.com/test/repo", input_shape=(1, 3, 32, 32),
    )


class TestBuildNodeDefs:
    """Tests for build_node_defs."""

    def test_produces_valid_structure(self) -> None:
        graph = _make_graph()
        defs = build_node_defs(graph)
        for _op_type, defn in defs.items():
            assert "category" in defn
            assert "headerColor" in defn
            assert "inputs" in defn
            assert "outputs" in defn
            assert "params" in defn

    def test_one_def_per_op_type(self) -> None:
        graph = _make_graph()
        defs = build_node_defs(graph)
        assert "Conv2d" in defs
        assert "ReLU" in defs
        assert "Linear" in defs

    def test_port_type_is_tensor(self) -> None:
        graph = _make_graph()
        defs = build_node_defs(graph)
        for defn in defs.values():
            for inp in defn["inputs"]:
                assert inp["type"] == "TENSOR"
            for out in defn["outputs"]:
                assert out["type"] == "TENSOR"

    def test_params_include_shapes(self) -> None:
        graph = _make_graph()
        defs = build_node_defs(graph)
        conv_params = defs["Conv2d"]["params"]
        param_keys = [p["key"] for p in conv_params]
        assert "_input_shape" in param_keys
        assert "_output_shape" in param_keys


class TestBuildInitialNodes:
    """Tests for build_initial_nodes."""

    def test_node_count(self) -> None:
        graph = _make_graph()
        nodes = build_initial_nodes(graph)
        assert len(nodes) == 3

    def test_has_positions(self) -> None:
        graph = _make_graph()
        nodes = build_initial_nodes(graph)
        for node in nodes:
            assert "x" in node
            assert "y" in node

    def test_ids_are_unique(self) -> None:
        graph = _make_graph()
        nodes = build_initial_nodes(graph)
        ids = [n["id"] for n in nodes]
        assert len(ids) == len(set(ids))

    def test_types_match(self) -> None:
        graph = _make_graph()
        nodes = build_initial_nodes(graph)
        types = {n["type"] for n in nodes}
        assert types == {"Conv2d", "ReLU", "Linear"}


class TestBuildInitialConnections:
    """Tests for build_initial_connections."""

    def test_connection_count(self) -> None:
        graph = _make_graph()
        connections = build_initial_connections(graph)
        assert len(connections) == 2

    def test_connection_structure(self) -> None:
        graph = _make_graph()
        connections = build_initial_connections(graph)
        for conn in connections:
            assert "id" in conn
            assert "fromNode" in conn
            assert "fromPort" in conn
            assert "toNode" in conn
            assert "toPort" in conn

    def test_port_indices_within_bounds(self) -> None:
        graph = _make_graph()
        node_defs = build_node_defs(graph)
        connections = build_initial_connections(graph)
        node_types = {n.id: n.op_type for n in graph.nodes}

        for conn in connections:
            source_type = node_types[conn["fromNode"]]
            target_type = node_types[conn["toNode"]]
            max_source_port = len(node_defs[source_type]["outputs"])
            max_target_port = len(node_defs[target_type]["inputs"])
            assert conn["fromPort"] < max_source_port
            assert conn["toPort"] < max_target_port
