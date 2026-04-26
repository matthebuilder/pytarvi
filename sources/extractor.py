import ast
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger
from sources.models import Node, Edge


class ArchitectureExtractor:
    """Extracts PyTorch architecture using static analysis of the AST."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self._model_class_name: Optional[str] = None

    def extract(self) -> Tuple[List[Node], List[Edge]]:
        """
        Analyzes the Python file and extracts nodes and edges.

        Returns:
            A tuple containing the list of extracted nodes and edges.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())

            # 1. Find the PyTorch model class
            model_class = self._find_model_class(tree)
            if not model_class:
                logger.error(
                    "No PyTorch model class (inheriting from nn.Module) found."
                )
                return [], []

            self._model_class_name = model_class.name
            logger.info(f"Analyzing model class: {self._model_class_name}")

            # 2. Extract layers from __init__
            self._extract_layers(model_class)

            # 3. Extract flow from forward
            self._extract_flow(model_class)

            return list(self.nodes.values()), self.edges

        except Exception as e:
            logger.exception(f"Error during architecture extraction: {e}")
            return [], []

    def _find_model_class(self, tree: ast.Module) -> Optional[ast.ClassDef]:
        """Finds a class that inherits from nn.Module."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for base in node.bases:
                    if isinstance(base, ast.Attribute) and base.attr == "Module":
                        return node
                    if isinstance(base, ast.Name) and base.id == "nn.Module":
                        return node
        return None

    def _extract_layers(self, model_class: ast.ClassDef) -> None:
        """Parses the __init__ method to identify layers."""
        init_method = next(
            (
                n
                for n in model_class.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ),
            None,
        )
        if not init_method:
            logger.warning("No __init__ method found in model class.")
            return

        for node in ast.walk(init_method):
            if isinstance(node, ast.Assign):
                # Looking for self.layer = nn.Something(...)
                if (
                    isinstance(node.targets[0], ast.Attribute)
                    and isinstance(node.targets[0].value, ast.Name)
                    and node.targets[0].value.id == "self"
                ):
                    layer_name = node.targets[0].attr
                    if isinstance(node.value, ast.Call):
                        # Determine layer type
                        layer_type = "Unknown"
                        if isinstance(node.value.func, ast.Attribute):
                            layer_type = node.value.func.attr
                        elif isinstance(node.value.func, ast.Name):
                            layer_type = node.value.func.id

                        self.nodes[layer_name] = Node(
                            id=layer_name,
                            name=layer_name,
                            type=layer_type,
                            params=self._parse_args(node.value),
                        )

    def _parse_args(self, call: ast.Call) -> Dict[str, Any]:
        """Simple parser for call arguments."""
        params = {}
        for i, arg in enumerate(call.args):
            if isinstance(arg, ast.Constant):
                params[f"arg_{i}"] = arg.value
            elif isinstance(arg, ast.Name):
                params[f"arg_{i}"] = arg.id
        for keyword in call.keywords:
            params[keyword.arg] = (
                ast.dump(keyword.value)
                if not isinstance(keyword.value, ast.Constant)
                else keyword.value.value
            )
        return params

    def _extract_flow(self, model_class: ast.ClassDef) -> None:
        """Parses the forward method to identify the flow between layers."""
        forward_method = next(
            (
                n
                for n in model_class.body
                if isinstance(n, ast.FunctionDef) and n.name == "forward"
            ),
            None,
        )
        if not forward_method:
            logger.warning("No forward method found in model class.")
            return

        # Track the last layer used to create edges between layers
        last_layer: Optional[str] = None

        # Iterate over the statements in the forward method to preserve order
        for stmt in forward_method.body:
            # Look for calls to self.layer within the statement
            for node in ast.walk(stmt):
                if isinstance(node, ast.Call):
                    if (
                        isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self"
                    ):
                        layer_name = node.func.attr
                        if layer_name in self.nodes:
                            if last_layer is not None:
                                edge_id = f"{last_layer}_{layer_name}"
                                self.edges.append(
                                    Edge(
                                        id=edge_id, source=last_layer, target=layer_name
                                    )
                                )
                            last_layer = layer_name
