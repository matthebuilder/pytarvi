"""Tests for sources.layout — topological sort and positioning."""



from sources.graph_model import ArchEdge, ArchNode
from sources.layout import compute_depth, compute_layout, topological_sort


def _make_node(nid: str, op_type: str = "ReLU") -> ArchNode:
    return ArchNode(
        id=nid, name=nid, op_type=op_type,
        category="Activation", color="#4ade80",
    )


class TestTopologicalSort:
    """Tests for topological_sort."""

    def test_linear_chain(self) -> None:
        nodes = [_make_node("a"), _make_node("b"), _make_node("c")]
        edges = [
            ArchEdge(id="e0", source_id="a", target_id="b"),
            ArchEdge(id="e1", source_id="b", target_id="c"),
        ]
        result = topological_sort(nodes, edges)
        assert result.index("a") < result.index("b")
        assert result.index("b") < result.index("c")

    def test_dag_with_branch(self) -> None:
        nodes = [
            _make_node("input"), _make_node("branch_a"),
            _make_node("branch_b"), _make_node("merge"),
        ]
        edges = [
            ArchEdge(id="e0", source_id="input", target_id="branch_a"),
            ArchEdge(id="e1", source_id="input", target_id="branch_b"),
            ArchEdge(id="e2", source_id="branch_a", target_id="merge"),
            ArchEdge(id="e3", source_id="branch_b", target_id="merge"),
        ]
        result = topological_sort(nodes, edges)
        assert result.index("input") < result.index("branch_a")
        assert result.index("input") < result.index("branch_b")
        assert result.index("branch_a") < result.index("merge")
        assert result.index("branch_b") < result.index("merge")

    def test_single_node(self) -> None:
        nodes = [_make_node("only")]
        result = topological_sort(nodes, [])
        assert result == ["only"]

    def test_empty_graph(self) -> None:
        result = topological_sort([], [])
        assert result == []

    def test_no_edges(self) -> None:
        nodes = [_make_node("x"), _make_node("y")]
        result = topological_sort(nodes, [])
        assert set(result) == {"x", "y"}


class TestComputeDepth:
    """Tests for compute_depth."""

    def test_linear_chain_depths(self) -> None:
        nodes = [_make_node("a"), _make_node("b"), _make_node("c")]
        edges = [
            ArchEdge(id="e0", source_id="a", target_id="b"),
            ArchEdge(id="e1", source_id="b", target_id="c"),
        ]
        depth = compute_depth(nodes, edges)
        assert depth["a"] == 0
        assert depth["b"] == 1
        assert depth["c"] == 2

    def test_branch_same_depth(self) -> None:
        nodes = [
            _make_node("input"), _make_node("a"),
            _make_node("b"), _make_node("out"),
        ]
        edges = [
            ArchEdge(id="e0", source_id="input", target_id="a"),
            ArchEdge(id="e1", source_id="input", target_id="b"),
            ArchEdge(id="e2", source_id="a", target_id="out"),
            ArchEdge(id="e3", source_id="b", target_id="out"),
        ]
        depth = compute_depth(nodes, edges)
        assert depth["input"] == 0
        assert depth["a"] == 1
        assert depth["b"] == 1
        assert depth["out"] == 2


class TestComputeLayout:
    """Tests for compute_layout."""

    def test_linear_positions(self) -> None:
        nodes = [_make_node("a"), _make_node("b")]
        edges = [ArchEdge(id="e0", source_id="a", target_id="b")]
        positions = compute_layout(nodes, edges)
        assert positions["a"][0] < positions["b"][0]

    def test_empty_graph(self) -> None:
        assert compute_layout([], []) == {}

    def test_single_node(self) -> None:
        nodes = [_make_node("solo")]
        positions = compute_layout(nodes, [])
        assert "solo" in positions
        assert positions["solo"][0] == 50.0
        assert positions["solo"][1] == 50.0

    def test_branch_layout(self) -> None:
        nodes = [_make_node("in"), _make_node("a"), _make_node("b"), _make_node("out")]
        edges = [
            ArchEdge(id="e0", source_id="in", target_id="a"),
            ArchEdge(id="e1", source_id="in", target_id="b"),
            ArchEdge(id="e2", source_id="a", target_id="out"),
            ArchEdge(id="e3", source_id="b", target_id="out"),
        ]
        positions = compute_layout(nodes, edges)
        assert positions["in"][0] < positions["a"][0]
        assert positions["a"][0] == positions["b"][0]
        assert positions["a"][0] < positions["out"][0]
        assert positions["a"][1] != positions["b"][1]
