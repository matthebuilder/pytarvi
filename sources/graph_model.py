"""Data model for the extracted architecture graph."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ArchNode:
    """A single node in the architecture graph representing one operation.

    Attributes:
        id: Unique node identifier (e.g. ``"conv1"``).
        name: Human-readable display name.
        op_type: Canonical operation type (e.g. ``"Conv2d"``).
        category: Node-editor category (e.g. ``"Convolution"``).
        input_shape: Shape of the input tensor, or ``None`` if unknown.
        output_shape: Shape of the output tensor, or ``None`` if unknown.
        params: Layer-specific parameters (kernel_size, out_channels, etc.).
        color: Hex header color for the node editor.
    """

    id: str
    name: str
    op_type: str
    category: str
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    color: str = "#6b7280"


@dataclass(frozen=True)
class ArchEdge:
    """A directed edge in the architecture graph representing data flow.

    Attributes:
        id: Unique edge identifier.
        source_id: ID of the source :class:`ArchNode`.
        source_port: Output port index on the source node.
        target_id: ID of the target :class:`ArchNode`.
        target_port: Input port index on the target node.
    """

    id: str
    source_id: str
    target_id: str
    source_port: int = 0
    target_port: int = 0


@dataclass
class ArchitectureGraph:
    """Complete architecture graph extracted from a PyTorch model.

    Attributes:
        nodes: Ordered list of :class:`ArchNode` instances.
        edges: List of :class:`ArchEdge` instances.
        model_name: Name of the model class.
        source_repo: GitHub URL of the source repository.
        input_shape: Input tensor shape used for tracing.
    """

    nodes: List[ArchNode] = field(default_factory=list)
    edges: List[ArchEdge] = field(default_factory=list)
    model_name: str = ""
    source_repo: str = ""
    input_shape: Tuple[int, ...] = ()

    def node_by_id(self, node_id: str) -> Optional[ArchNode]:
        """Look up a node by its ID.

        Args:
            node_id: The unique node identifier.

        Returns:
            The matching :class:`ArchNode`, or ``None`` if not found.
        """
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def edges_from(self, node_id: str) -> List[ArchEdge]:
        """Return all outgoing edges from a given node.

        Args:
            node_id: The source node identifier.

        Returns:
            List of :class:`ArchEdge` instances originating from the node.
        """
        return [e for e in self.edges if e.source_id == node_id]

    def edges_to(self, node_id: str) -> List[ArchEdge]:
        """Return all incoming edges to a given node.

        Args:
            node_id: The target node identifier.

        Returns:
            List of :class:`ArchEdge` instances targeting the node.
        """
        return [e for e in self.edges if e.target_id == node_id]

    def unique_op_types(self) -> List[str]:
        """Return unique operation types present in the graph.

        Returns:
            Sorted list of unique ``op_type`` strings.
        """
        return sorted({n.op_type for n in self.nodes})
