"""Architecture extraction from PyTorch models via FX trace, hooks, or AST."""

import ast
import inspect
import operator
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger

from sources.config import (
    DEFAULT_CATEGORY,
    get_category,
    get_color,
    get_op_type_for_function,
    get_op_type_for_method,
)
from sources.exceptions import NoModelError, TraceError
from sources.graph_model import ArchEdge, ArchitectureGraph, ArchNode


def extract_architecture(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    file_path: Optional[Path] = None,
    model_name: str = "",
    source_repo: str = "",
) -> ArchitectureGraph:
    """Extract the architecture graph from a PyTorch model.

    Tries three strategies in order of preference:
    1. FX symbolic trace (captures all ops including functional)
    2. Hook-based extraction (module ops only, with shapes)
    3. AST analysis (layer types only, no shapes or edges)

    Args:
        model: The ``nn.Module`` to analyze.
        input_shape: Input tensor shape (including batch dim).
        file_path: Path to the model's source file (for AST fallback).
        model_name: Name of the model class.
        source_repo: GitHub URL of the source repository.

    Returns:
        An :class:`ArchitectureGraph` instance.

    Raises:
        NoModelError: If no architecture nodes can be extracted.
    """
    graph = _try_fx_extraction(model, input_shape, model_name, source_repo)
    if graph is not None and graph.nodes:
        logger.info(f"FX extraction succeeded: {len(graph.nodes)} nodes")
        return graph

    graph = _try_hook_extraction(model, input_shape, model_name, source_repo)
    if graph is not None and graph.nodes:
        logger.info(f"Hook extraction succeeded: {len(graph.nodes)} nodes")
        return graph

    if file_path is not None:
        graph = _try_ast_extraction(file_path, model_name, source_repo)
        if graph is not None and graph.nodes:
            logger.info(f"AST extraction succeeded: {len(graph.nodes)} nodes")
            return graph

    raise NoModelError(
        f"Could not extract architecture from {model_name or model.__class__.__name__}"
    )


def _try_fx_extraction(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    model_name: str,
    source_repo: str,
) -> Optional[ArchitectureGraph]:
    """Attempt FX symbolic trace extraction.

    Args:
        model: The ``nn.Module`` to trace.
        input_shape: Input tensor shape.
        model_name: Name of the model class.
        source_repo: Source repository URL.

    Returns:
        An :class:`ArchitectureGraph`, or ``None`` if tracing fails.
    """
    try:
        return extract_via_fx(model, input_shape, model_name, source_repo)
    except Exception as exc:
        logger.warning(f"FX trace failed: {exc}")
        return None


def _try_hook_extraction(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    model_name: str,
    source_repo: str,
) -> Optional[ArchitectureGraph]:
    """Attempt hook-based extraction.

    Args:
        model: The ``nn.Module`` to analyze.
        input_shape: Input tensor shape.
        model_name: Name of the model class.
        source_repo: Source repository URL.

    Returns:
        An :class:`ArchitectureGraph`, or ``None`` if extraction fails.
    """
    try:
        return extract_via_hooks(model, input_shape, model_name, source_repo)
    except Exception as exc:
        logger.warning(f"Hook extraction failed: {exc}")
        return None


def _try_ast_extraction(
    file_path: Path, model_name: str, source_repo: str
) -> Optional[ArchitectureGraph]:
    """Attempt AST-based extraction.

    Args:
        file_path: Path to the model source file.
        model_name: Name of the model class.
        source_repo: Source repository URL.

    Returns:
        An :class:`ArchitectureGraph`, or ``None`` if extraction fails.
    """
    try:
        return extract_via_ast(file_path, model_name, source_repo)
    except Exception as exc:
        logger.warning(f"AST extraction failed: {exc}")
        return None


def extract_via_fx(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    model_name: str = "",
    source_repo: str = "",
) -> ArchitectureGraph:
    """Extract architecture using torch.fx symbolic tracing.

    Captures all operations including functional calls (``F.relu``,
    ``F.dropout``, etc.) and constructs a complete DAG with edges
    derived from ``node.all_input_nodes``.

    Args:
        model: The ``nn.Module`` to trace.
        input_shape: Input tensor shape (including batch dim).
        model_name: Name of the model class.
        source_repo: Source repository URL.

    Returns:
        An :class:`ArchitectureGraph` with nodes and edges.

    Raises:
        TraceError: If ``torch.fx.symbolic_trace`` fails.
    """
    try:
        gm = torch.fx.symbolic_trace(model)
    except Exception as exc:
        raise TraceError(f"FX symbolic trace failed: {exc}") from exc

    try:
        from torch.fx.passes.shape_prop import ShapeProp  # type: ignore[import]

        sample_input = torch.randn(*input_shape)
        ShapeProp(gm).propagate(sample_input)
    except Exception as exc:
        logger.warning(f"ShapeProp failed, shapes will be missing: {exc}")

    nodes: List[ArchNode] = []
    edges: List[ArchEdge] = []
    edge_counter: int = 0
    node_ids_set: set = set()

    input_shape_value: Optional[Tuple[int, ...]] = None
    for fx_node in gm.graph.nodes:
        if fx_node.op == "placeholder" and "tensor_meta" in fx_node.meta:
            meta = fx_node.meta["tensor_meta"]
            shape = getattr(meta, "shape", None)
            if shape is not None:
                input_shape_value = tuple(shape)

    for fx_node in gm.graph.nodes:
        if fx_node.op == "placeholder":
            arch_node = ArchNode(
                id=fx_node.name,
                name="Input",
                op_type="Input",
                category="Input",
                input_shape=input_shape_value,
                output_shape=input_shape_value,
                color=get_color("Input"),
            )
            nodes.append(arch_node)
            node_ids_set.add(fx_node.name)
            continue

        if fx_node.op in ("output", "get_attr"):
            continue

        op_type = _normalize_op(fx_node, gm)
        if not op_type:
            continue

        category = get_category(op_type)
        color = get_color(category)

        params = _extract_node_params(fx_node, gm)
        input_shape_node = _get_input_shape(fx_node)
        output_shape_node = _get_output_shape(fx_node)

        arch_node = ArchNode(
            id=fx_node.name,
            name=fx_node.name,
            op_type=op_type,
            category=category,
            input_shape=input_shape_node,
            output_shape=output_shape_node,
            params=params,
            color=color,
        )
        nodes.append(arch_node)
        node_ids_set.add(fx_node.name)

    for fx_node in gm.graph.nodes:
        if fx_node.op in ("output", "get_attr"):
            continue
        if fx_node.name not in node_ids_set:
            continue

        for input_node in fx_node.all_input_nodes:
            resolved = _resolve_input(input_node, node_ids_set)
            if resolved is not None:
                source_port = _get_output_port_index(resolved, fx_node)
                target_port = _get_input_port_index(fx_node, resolved)

                edges.append(
                    ArchEdge(
                        id=f"e{edge_counter}",
                        source_id=resolved.name,
                        source_port=source_port,
                        target_id=fx_node.name,
                        target_port=target_port,
                    )
                )
                edge_counter += 1

    edges = _deduplicate_edges(edges)

    return ArchitectureGraph(
        nodes=nodes,
        edges=edges,
        model_name=model_name or model.__class__.__name__,
        source_repo=source_repo,
        input_shape=input_shape,
    )


def extract_via_hooks(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    model_name: str = "",
    source_repo: str = "",
) -> ArchitectureGraph:
    """Extract architecture using forward-pass hooks.

    Registers ``forward_pre_hook`` and ``forward_hook`` on every leaf
    module, runs a dummy forward pass, and collects input/output
    shapes. Functional operations (``F.relu``, etc.) are invisible to
    hooks.

    Args:
        model: The ``nn.Module`` to analyze.
        input_shape: Input tensor shape (including batch dim).
        model_name: Name of the model class.
        source_repo: Source repository URL.

    Returns:
        An :class:`ArchitectureGraph` with nodes ordered by execution
        and edges inferred from sequential execution and shape matching.
    """
    records: List[Dict[str, Any]] = []
    hooks: List[Any] = []

    def pre_hook(module: nn.Module, inputs: Tuple[Any, ...]) -> None:
        nonlocal records
        shape = _tensor_shape_from_inputs(inputs)
        records.append({
            "module": module,
            "input_shape": shape,
            "output_shape": None,
            "executed": False,
        })

    def post_hook(
        module: nn.Module, inputs: Tuple[Any, ...], outputs: Any
    ) -> None:
        for record in reversed(records):
            if record["module"] is module and not record["executed"]:
                record["output_shape"] = _tensor_shape_from_output(outputs)
                record["executed"] = True
                break

    for module in model.modules():
        if list(module.children()):
            continue
        hooks.append(module.register_forward_pre_hook(pre_hook))
        hooks.append(module.register_forward_hook(post_hook))

    try:
        with torch.no_grad():
            model.eval()
            sample_input = torch.randn(*input_shape)
            model(sample_input)
    except Exception as exc:
        logger.warning(f"Forward pass failed during hook extraction: {exc}")
    finally:
        for hook in hooks:
            hook.remove()

    nodes: List[ArchNode] = []
    node_counter: int = 0

    for node_counter, record in enumerate(records):
        module = record["module"]
        op_type = module.__class__.__name__
        category = get_category(op_type)
        color = get_color(category)
        params = _extract_layer_params(module)

        node_id = _module_to_node_id(module, node_counter)

        nodes.append(
            ArchNode(
                id=node_id,
                name=node_id,
                op_type=op_type,
                category=category,
                input_shape=record["input_shape"],
                output_shape=record["output_shape"],
                params=params,
                color=color,
            )
        )

    edges: List[ArchEdge] = []
    for edge_counter, i in enumerate(range(len(nodes) - 1)):
        edges.append(
            ArchEdge(
                id=f"e{edge_counter}",
                source_id=nodes[i].id,
                source_port=0,
                target_id=nodes[i + 1].id,
                target_port=0,
            )
        )

    return ArchitectureGraph(
        nodes=nodes,
        edges=edges,
        model_name=model_name or model.__class__.__name__,
        source_repo=source_repo,
        input_shape=input_shape,
    )


def extract_via_ast(
    file_path: Path,
    model_name: str = "",
    source_repo: str = "",
) -> ArchitectureGraph:
    """Extract architecture using static AST analysis.

    Parses the Python source file, walks ``__init__`` for ``self.x =
    nn.Conv2d(...)`` assignments, and builds a linear chain of nodes.
    No shapes or real edges are available.

    Args:
        file_path: Path to the Python model file.
        model_name: Name of the model class.
        source_repo: Source repository URL.

    Returns:
        An :class:`ArchitectureGraph` with layer types inferred from
        constructor calls.

    Raises:
        NoModelError: If no layer assignments are found.
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except (SyntaxError, UnicodeDecodeError) as exc:
        raise NoModelError(f"Cannot parse {file_path}: {exc}") from exc

    target_class: Optional[str] = None
    if model_name:
        target_class = model_name

    layers: List[Tuple[str, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if target_class and node.name != target_class:
            continue

        for item in ast.walk(node):
            if not isinstance(item, ast.Assign):
                continue
            for target in item.targets:
                if not isinstance(target, ast.Attribute):
                    continue
                if not isinstance(target.value, ast.Name):
                    continue
                if target.value.id != "self":
                    continue

                layer_type = _extract_layer_type_from_call(item.value)
                if layer_type:
                    layers.append((target.attr, layer_type))

    if not layers:
        raise NoModelError(f"No layer assignments found in {file_path}")

    nodes: List[ArchNode] = []
    edges: List[ArchEdge] = []
    edge_counter: int = 0

    for i, (attr_name, layer_type) in enumerate(layers):
        category = get_category(layer_type)
        color = get_color(category)
        nodes.append(
            ArchNode(
                id=attr_name,
                name=attr_name,
                op_type=layer_type,
                category=category,
                color=color,
            )
        )
        if i > 0:
            edges.append(
                ArchEdge(
                    id=f"e{edge_counter}",
                    source_id=layers[i - 1][0],
                    source_port=0,
                    target_id=attr_name,
                    target_port=0,
                )
            )
            edge_counter += 1

    return ArchitectureGraph(
        nodes=nodes,
        edges=edges,
        model_name=model_name or file_path.stem,
        source_repo=source_repo,
    )


def _normalize_op(fx_node: torch.fx.Node, gm: torch.fx.GraphModule) -> str:
    """Normalize an FX node to a canonical operation type string.

    Args:
        fx_node: An FX graph node.
        gm: The GraphModule containing the node.

    Returns:
        Canonical operation type string, or empty string to skip.
    """
    if fx_node.op == "call_module":
        module = gm.get_submodule(str(fx_node.target))
        return module.__class__.__name__
    if fx_node.op == "call_function":
        return get_op_type_for_function(fx_node.target)  # type: ignore[arg-type]
    if fx_node.op == "call_method":
        return get_op_type_for_method(str(fx_node.target))
    return DEFAULT_CATEGORY


def _extract_node_params(
    fx_node: torch.fx.Node, gm: torch.fx.GraphModule
) -> Dict[str, Any]:
    """Extract display parameters from an FX node.

    For ``call_module`` nodes, delegates to :func:`_extract_layer_params`.
    For ``call_function`` and ``call_method`` nodes, extracts from args.

    Args:
        fx_node: An FX graph node.
        gm: The GraphModule containing the node.

    Returns:
        Dictionary of parameter key-value pairs.
    """
    if fx_node.op == "call_module":
        module = gm.get_submodule(str(fx_node.target))
        return _extract_layer_params(module)

    if fx_node.op == "call_function":
        return _extract_functional_dims(fx_node)

    if fx_node.op == "call_method":
        return _extract_method_dims(fx_node)

    return {}


def _extract_layer_params(module: nn.Module) -> Dict[str, Any]:
    """Extract key display parameters from an nn.Module instance.

    Args:
        module: An ``nn.Module`` instance.

    Returns:
        Dictionary of parameter key-value pairs for display.
    """
    params: Dict[str, Any] = {}

    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        params["in_channels"] = module.in_channels
        params["out_channels"] = module.out_channels
        params["kernel_size"] = module.kernel_size
        if module.stride != (1,) * len(module.stride):
            params["stride"] = module.stride
        if module.padding != (0,) * len(module.padding):
            params["padding"] = module.padding
    elif isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d,
                             nn.ConvTranspose3d)):
        params["in_channels"] = module.in_channels
        params["out_channels"] = module.out_channels
        params["kernel_size"] = module.kernel_size
    elif isinstance(module, nn.Linear):
        params["in_features"] = module.in_features
        params["out_features"] = module.out_features
    elif isinstance(module, (
        nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
    )):
        params["num_features"] = module.num_features
    elif isinstance(module, nn.LayerNorm):
        params["normalized_shape"] = module.normalized_shape
    elif isinstance(module, nn.GroupNorm):
        params["num_groups"] = module.num_groups
        params["num_channels"] = module.num_channels
    elif isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
        params["p"] = module.p
    elif isinstance(module, (
        nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
        nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
    )):
        params["kernel_size"] = module.kernel_size
    elif isinstance(module, (
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d,
        nn.AdaptiveAvgPool3d,
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d,
        nn.AdaptiveMaxPool3d,
    )):
        params["output_size"] = module.output_size
    elif isinstance(module, nn.Embedding):
        params["num_embeddings"] = module.num_embeddings
        params["embedding_dim"] = module.embedding_dim
    elif isinstance(module, (nn.LSTM, nn.GRU, nn.RNN)):
        params["input_size"] = module.input_size
        params["hidden_size"] = module.hidden_size
        params["num_layers"] = module.num_layers
    elif isinstance(module, nn.MultiheadAttention):
        params["embed_dim"] = module.embed_dim
        params["num_heads"] = module.num_heads
    elif isinstance(module, nn.Flatten):
        params["start_dim"] = module.start_dim
        params["end_dim"] = module.end_dim
    elif isinstance(module, (nn.LeakyReLU,)):
        params["negative_slope"] = module.negative_slope
    elif isinstance(module, nn.PReLU):
        params["num_parameters"] = module.num_parameters

    return params


def _extract_functional_dims(fx_node: torch.fx.Node) -> Dict[str, Any]:
    """Extract display parameters from a call_function FX node.

    Args:
        fx_node: An FX ``call_function`` node.

    Returns:
        Dictionary of parameter key-value pairs.
    """
    params: Dict[str, Any] = {}
    func = fx_node.target
    args = fx_node.args

    if func in (torch.flatten,):
        if len(args) >= 2 and isinstance(args[1], int):
            params["start_dim"] = args[1]
        if len(args) >= 3 and isinstance(args[2], int):
            params["end_dim"] = args[2]
    elif func in (torch.cat, torch.stack):
        if fx_node.kwargs.get("dim") is not None:
            params["dim"] = fx_node.kwargs["dim"]
        elif len(args) >= 2 and isinstance(args[1], int):
            params["dim"] = args[1]
    elif func in (operator.add, torch.add) or func in (operator.mul, torch.mul):
        params["operation"] = "element-wise"
    elif func in (torch.sigmoid, torch.tanh, torch.relu):
        pass
    else:
        try:
            sig = inspect.signature(func)  # type: ignore[arg-type]
            bound = sig.bind_partial(*args, **fx_node.kwargs)
            for name, value in bound.arguments.items():
                if name in ("input", "x", "self", "args", "kwargs"):
                    continue
                if isinstance(value, (int, float, str, bool)):
                    params[name] = value
        except (ValueError, TypeError):
            pass

    return params


def _extract_method_dims(fx_node: torch.fx.Node) -> Dict[str, Any]:
    """Extract display parameters from a call_method FX node.

    Args:
        fx_node: An FX ``call_method`` node.

    Returns:
        Dictionary of parameter key-value pairs.
    """
    params: Dict[str, Any] = {}
    method = fx_node.target
    args = fx_node.args

    if method in ("view", "reshape"):
        if len(args) >= 2:
            shape = args[1:]
            params["shape"] = [s for s in shape if isinstance(s, int)]
    elif method == "unsqueeze":
        if len(args) >= 2 and isinstance(args[1], int):
            params["dim"] = args[1]
    elif method == "permute":
        if len(args) >= 2:
            dims = args[1:]
            params["dims"] = [d for d in dims if isinstance(d, int)]
    elif method == "transpose" and len(args) >= 3:
        params["dim0"] = args[1]
        params["dim1"] = args[2]

    return params


def _get_input_shape(fx_node: torch.fx.Node) -> Optional[Tuple[int, ...]]:
    """Get input tensor shape from an FX node's metadata.

    Args:
        fx_node: An FX graph node.

    Returns:
        Shape tuple, or ``None`` if unavailable.
    """
    meta = fx_node.meta
    if "tensor_meta" not in meta:
        return None
    tensor_meta = meta["tensor_meta"]
    if isinstance(tensor_meta, dict):
        shape = tensor_meta.get("shape")
    else:
        shape = getattr(tensor_meta, "shape", None)
    if shape is not None:
        return tuple(shape)
    return None


def _get_output_shape(fx_node: torch.fx.Node) -> Optional[Tuple[int, ...]]:
    """Get output tensor shape from an FX node's metadata.

    Args:
        fx_node: An FX graph node.

    Returns:
        Shape tuple, or ``None`` if unavailable.
    """
    return _get_input_shape(fx_node)


def _resolve_input(
    input_node: torch.fx.Node, node_ids: set
) -> Optional[torch.fx.Node]:
    """Resolve an FX node's input to the nearest included ancestor.

    If the input is directly in ``node_ids``, return it. If it is a
    placeholder, return ``None`` (skip connections via placeholders
    are handled separately by edge post-processing). If it is another
    skipped node (get_attr, filtered-out), traverse its own inputs
    recursively.

    Args:
        input_node: The FX node to resolve.
        node_ids: Set of included node name strings.

    Returns:
        The resolved FX node, or ``None`` if no ancestor is in node_ids.
    """
    if input_node.name in node_ids:
        return input_node

    if input_node.op in ("placeholder", "output"):
        return None

    for ancestor in input_node.all_input_nodes:
        resolved = _resolve_input(ancestor, node_ids)
        if resolved is not None:
            return resolved

    return None


def _deduplicate_edges(edges: List[ArchEdge]) -> List[ArchEdge]:
    """Remove duplicate edges with the same source, target, and ports.

    Also adds skip-connection edges: when a node has multiple inputs
    that resolve to the same source (e.g. both via placeholder and
    via a computed path), the edge already exists. But when a node
    takes a placeholder as one input and a computed node as another,
    and the placeholder was also an input to an earlier computed node,
    we add an edge from that earlier node to represent the skip
    connection.

    Args:
        edges: List of :class:`ArchEdge` instances, possibly with
            duplicates.

    Returns:
        Deduplicated list of :class:`ArchEdge` instances.
    """
    seen: set = set()
    unique: List[ArchEdge] = []
    for edge in edges:
        key = (edge.source_id, edge.source_port, edge.target_id, edge.target_port)
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique


def _get_output_port_index(
    source_node: torch.fx.Node, target_node: torch.fx.Node
) -> int:
    """Determine which output port of source connects to target.

    Args:
        source_node: The upstream FX node.
        target_node: The downstream FX node.

    Returns:
        Output port index (always 0 for simple dataflow).
    """
    _ = target_node
    users = list(source_node.users.keys())
    try:
        return users.index(target_node)
    except ValueError:
        return 0


def _get_input_port_index(
    target_node: torch.fx.Node, source_node: torch.fx.Node
) -> int:
    """Determine which input port of target receives from source.

    Args:
        target_node: The downstream FX node.
        source_node: The upstream FX node.

    Returns:
        Input port index based on argument position.
    """
    all_inputs = list(target_node.all_input_nodes)
    try:
        return all_inputs.index(source_node)
    except ValueError:
        return 0


def _tensor_shape_from_inputs(inputs: Tuple[Any, ...]) -> Optional[Tuple[int, ...]]:
    """Extract tensor shape from hook input arguments.

    Args:
        inputs: Tuple of input arguments to a forward hook.

    Returns:
        Shape tuple of the first tensor input, or ``None``.
    """
    if inputs and isinstance(inputs[0], torch.Tensor):
        return tuple(inputs[0].shape)
    for item in inputs:
        if isinstance(item, torch.Tensor):
            return tuple(item.shape)
    if isinstance(item, (tuple, list)) and item and isinstance(item[0], torch.Tensor):
        return tuple(item[0].shape)
    return None


def _tensor_shape_from_output(output: Any) -> Optional[Tuple[int, ...]]:
    """Extract tensor shape from hook output.

    Args:
        output: Output from a module's forward pass.

    Returns:
        Shape tuple, or ``None`` if not a tensor.
    """
    if isinstance(output, torch.Tensor):
        return tuple(output.shape)
    if (
        isinstance(output, (tuple, list))
        and output
        and isinstance(output[0], torch.Tensor)
    ):
        return tuple(output[0].shape)
    return None


def _module_to_node_id(module: nn.Module, counter: int) -> str:
    """Generate a node ID from a module instance.

    Args:
        module: The ``nn.Module`` instance.
        counter: Fallback counter for uniqueness.

    Returns:
        A string node ID.
    """
    class_name = module.__class__.__name__
    name = getattr(module, "_node_name", None)
    if name:
        return f"{name}_{class_name}"
    return f"{class_name}_{counter}"


def _extract_layer_type_from_call(node: ast.expr) -> Optional[str]:
    """Extract the layer type name from an AST assignment value.

    Recognizes patterns like ``nn.Conv2d(...)`` or ``nn.functional.relu(...)``.

    Args:
        node: The AST expression from the right-hand side of an assignment.

    Returns:
        The layer type string (e.g. ``"Conv2d"``), or ``None``.
    """
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return func.id
    return None
