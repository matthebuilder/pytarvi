from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple


@dataclass
class Node:
    """Represents a layer or functional block in a PyTorch architecture."""

    id: str
    name: str
    type: str
    params: Dict[str, Any] = field(default_factory=dict)
    input_dim: Optional[Any] = None
    output_dim: Optional[Any] = None
    position: Tuple[float, float] = (0.0, 0.0)


@dataclass
class Edge:
    """Represents the data flow between two nodes in the architecture."""

    id: str
    source: str
    target: str
    label: Optional[str] = None
