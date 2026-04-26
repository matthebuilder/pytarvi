"""Auto-layout algorithm for the node editor canvas."""

from collections import defaultdict
from typing import Dict, List, Tuple

from sources.graph_model import ArchEdge, ArchNode

X_SPACING: float = 300.0
Y_SPACING: float = 150.0
Y_OFFSET: float = 50.0
X_OFFSET: float = 50.0


def topological_sort(nodes: List[ArchNode], edges: List[ArchEdge]) -> List[str]:
    """Perform a topological sort of the architecture DAG.

    Uses Kahn's algorithm to produce a valid linear ordering of nodes
    such that for every edge (u, v), node u appears before node v.

    Args:
        nodes: List of :class:`ArchNode` instances.
        edges: List of :class:`ArchEdge` instances.

    Returns:
        Ordered list of node IDs. Nodes with no incoming edges appear
        first. If the graph contains a cycle, remaining nodes are
        appended in arbitrary order.
    """
    in_degree: Dict[str, int] = {n.id: 0 for n in nodes}
    adjacency: Dict[str, List[str]] = defaultdict(list)

    for edge in edges:
        adjacency[edge.source_id].append(edge.target_id)
        in_degree[edge.target_id] = in_degree.get(edge.target_id, 0) + 1

    queue: List[str] = [nid for nid, deg in in_degree.items() if deg == 0]
    result: List[str] = []

    while queue:
        current = queue.pop(0)
        result.append(current)
        for neighbor in adjacency[current]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    for node in nodes:
        if node.id not in result:
            result.append(node.id)

    return result


def compute_depth(
    nodes: List[ArchNode], edges: List[ArchEdge]
) -> Dict[str, int]:
    """Compute the depth (longest path from input) for each node.

    Args:
        nodes: List of :class:`ArchNode` instances.
        edges: List of :class:`ArchEdge` instances.

    Returns:
        Mapping of node ID to depth (0-based). Nodes with no
        incoming edges get depth 0.
    """
    incoming: Dict[str, List[str]] = defaultdict(list)
    for edge in edges:
        incoming[edge.target_id].append(edge.source_id)

    node_ids = {n.id for n in nodes}
    depth: Dict[str, int] = {}

    sorted_ids = topological_sort(nodes, edges)
    for nid in sorted_ids:
        if nid not in node_ids:
            continue
        predecessors = [p for p in incoming[nid] if p in depth]
        if predecessors:
            depth[nid] = max(depth[p] for p in predecessors) + 1
        else:
            depth[nid] = 0

    return depth


def compute_layout(
    nodes: List[ArchNode], edges: List[ArchEdge]
) -> Dict[str, Tuple[float, float]]:
    """Compute (x, y) positions for each node in the architecture graph.

    Nodes are arranged in a left-to-right dataflow layout. Depth is
    determined by longest path from input nodes. Within each depth
    column, nodes are spread vertically with even spacing.

    Args:
        nodes: List of :class:`ArchNode` instances.
        edges: List of :class:`ArchEdge` instances.

    Returns:
        Mapping of node ID to ``(x, y)`` position tuple.
    """
    if not nodes:
        return {}

    depth = compute_depth(nodes, edges)

    columns: Dict[int, List[str]] = defaultdict(list)
    for node in nodes:
        columns[depth.get(node.id, 0)].append(node.id)

    positions: Dict[str, Tuple[float, float]] = {}
    for col_idx, col_nodes in columns.items():
        x = X_OFFSET + col_idx * X_SPACING
        for row_idx, nid in enumerate(col_nodes):
            y = Y_OFFSET + row_idx * Y_SPACING
            positions[nid] = (x, y)

    return positions
