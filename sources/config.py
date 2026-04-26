"""Constants and mappings for the PyTorch Architecture Viewer."""

import operator
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F  # noqa: N812

PORT_TYPE: str = "TENSOR"

NODE_COLORS: Dict[str, str] = {
    "Convolution": "#3b82f6",
    "Normalization": "#facc15",
    "Activation": "#4ade80",
    "Pooling": "#a78bfa",
    "Dropout": "#fb923c",
    "Linear": "#60a5fa",
    "Merge": "#f87171",
    "Reshape": "#94a3b8",
    "Input": "#22d3ee",
    "Output": "#ec4899",
    "Embedding": "#c084fc",
    "RNN": "#f472b6",
    "Attention": "#fbbf24",
    "Loss": "#ef4444",
    "Other": "#6b7280",
}

CATEGORY_MAP: Dict[str, str] = {
    "Conv1d": "Convolution",
    "Conv2d": "Convolution",
    "Conv3d": "Convolution",
    "ConvTranspose1d": "Convolution",
    "ConvTranspose2d": "Convolution",
    "ConvTranspose3d": "Convolution",
    "BatchNorm1d": "Normalization",
    "BatchNorm2d": "Normalization",
    "BatchNorm3d": "Normalization",
    "InstanceNorm1d": "Normalization",
    "InstanceNorm2d": "Normalization",
    "InstanceNorm3d": "Normalization",
    "LayerNorm": "Normalization",
    "GroupNorm": "Normalization",
    "SyncBatchNorm": "Normalization",
    "ReLU": "Activation",
    "LeakyReLU": "Activation",
    "PReLU": "Activation",
    "ELU": "Activation",
    "SELU": "Activation",
    "GELU": "Activation",
    "SiLU": "Activation",
    "Mish": "Activation",
    "Softmax": "Activation",
    "Sigmoid": "Activation",
    "Tanh": "Activation",
    "Hardswish": "Activation",
    "Hardsigmoid": "Activation",
    "MaxPool1d": "Pooling",
    "MaxPool2d": "Pooling",
    "MaxPool3d": "Pooling",
    "AvgPool1d": "Pooling",
    "AvgPool2d": "Pooling",
    "AvgPool3d": "Pooling",
    "AdaptiveAvgPool1d": "Pooling",
    "AdaptiveAvgPool2d": "Pooling",
    "AdaptiveAvgPool3d": "Pooling",
    "AdaptiveMaxPool1d": "Pooling",
    "AdaptiveMaxPool2d": "Pooling",
    "AdaptiveMaxPool3d": "Pooling",
    "Dropout": "Dropout",
    "Dropout2d": "Dropout",
    "Dropout3d": "Dropout",
    "AlphaDropout": "Dropout",
    "Linear": "Linear",
    "Bilinear": "Linear",
    "LazyLinear": "Linear",
    "Add": "Merge",
    "Concat": "Merge",
    "Mul": "Merge",
    "Flatten": "Reshape",
    "Reshape": "Reshape",
    "View": "Reshape",
    "Unflatten": "Reshape",
    "Embedding": "Embedding",
    "EmbeddingBag": "Embedding",
    "LSTM": "RNN",
    "GRU": "RNN",
    "RNN": "RNN",
    "RNNBase": "RNN",
    "MultiheadAttention": "Attention",
    "CrossEntropyLoss": "Loss",
    "MSELoss": "Loss",
    "BCELoss": "Loss",
    "BCEWithLogitsLoss": "Loss",
    "NLLLoss": "Loss",
}

FUNCTIONAL_TO_MODULE_MAP: Dict[Callable, str] = {
    F.relu: "ReLU",
    F.leaky_relu: "LeakyReLU",
    F.elu: "ELU",
    F.selu: "SELU",
    F.gelu: "GELU",
    F.silu: "SiLU",
    F.mish: "Mish",
    F.hardswish: "Hardswish",
    F.hardsigmoid: "Hardsigmoid",
    F.softmax: "Softmax",
    F.sigmoid: "Sigmoid",
    F.tanh: "Tanh",
    F.dropout: "Dropout",
    F.dropout2d: "Dropout2d",
    F.dropout3d: "Dropout3d",
    F.batch_norm: "BatchNorm",
    F.layer_norm: "LayerNorm",
    F.group_norm: "GroupNorm",
    F.conv1d: "Conv1d",
    F.conv2d: "Conv2d",
    F.conv3d: "Conv3d",
    F.conv_transpose2d: "ConvTranspose2d",
    F.conv_transpose3d: "ConvTranspose3d",
    F.linear: "Linear",
    F.bilinear: "Bilinear",
    F.max_pool1d: "MaxPool1d",
    F.max_pool2d: "MaxPool2d",
    F.max_pool3d: "MaxPool3d",
    F.avg_pool1d: "AvgPool1d",
    F.avg_pool2d: "AvgPool2d",
    F.avg_pool3d: "AvgPool3d",
    F.adaptive_avg_pool1d: "AdaptiveAvgPool1d",
    F.adaptive_avg_pool2d: "AdaptiveAvgPool2d",
    F.adaptive_avg_pool3d: "AdaptiveAvgPool3d",
    F.adaptive_max_pool1d: "AdaptiveMaxPool1d",
    F.adaptive_max_pool2d: "AdaptiveMaxPool2d",
    F.adaptive_max_pool3d: "AdaptiveMaxPool3d",
    F.embedding: "Embedding",
    F.multi_head_attention_forward: "MultiheadAttention",
    torch.relu: "ReLU",
    torch.relu_: "ReLU",
    torch.flatten: "Flatten",
    torch.cat: "Concat",
    torch.stack: "Concat",
    torch.add: "Add",
    torch.mul: "Mul",
    torch.sigmoid: "Sigmoid",
    torch.tanh: "Tanh",
    torch.softmax: "Softmax",
    operator.add: "Add",
    operator.mul: "Mul",
    operator.iadd: "Add",
    operator.imul: "Mul",
}

METHOD_TO_OP_MAP: Dict[str, str] = {
    "view": "View",
    "reshape": "Reshape",
    "flatten": "Flatten",
    "contiguous": "",
    "unsqueeze": "Reshape",
    "squeeze": "Reshape",
    "permute": "Reshape",
    "transpose": "Reshape",
    "expand": "Reshape",
    "repeat": "Reshape",
    "split": "Reshape",
    "chunk": "Reshape",
}

COMMON_INPUT_SHAPES: Dict[str, Tuple[int, ...]] = {
    "vision_cnn": (1, 3, 224, 224),
    "vision_cnn_small": (1, 3, 32, 32),
    "nlp_transformer": (1, 128),
    "nlp_rnn": (1, 64, 128),
}

DEFAULT_HEIGHT: int = 700

DEFAULT_CATEGORY: str = "Other"


def get_category(op_type: str) -> str:
    """Return the node-editor category for a given operation type.

    Args:
        op_type: Canonical operation type string (e.g. ``"Conv2d"``).

    Returns:
        Category string from :data:`CATEGORY_MAP`, or :data:`DEFAULT_CATEGORY`.
    """
    return CATEGORY_MAP.get(op_type, DEFAULT_CATEGORY)


def get_color(category: str) -> str:
    """Return the hex header color for a node-editor category.

    Args:
        category: Category string (e.g. ``"Convolution"``).

    Returns:
        Hex color string from :data:`NODE_COLORS`, or gray ``#6b7280``.
    """
    return NODE_COLORS.get(category, NODE_COLORS[DEFAULT_CATEGORY])


def get_op_type_for_function(func: Callable) -> str:
    """Normalize a functional call to its canonical operation type.

    Args:
        func: The callable used in a ``call_function`` FX node.

    Returns:
        Canonical operation type string, or the function's ``__name__``.
    """
    return FUNCTIONAL_TO_MODULE_MAP.get(func, getattr(func, "__name__", str(func)))


def get_op_type_for_method(method_name: str) -> str:
    """Normalize a method call to its canonical operation type.

    Args:
        method_name: The method name from a ``call_method`` FX node.

    Returns:
        Canonical operation type string, or empty string if the method
        should be skipped (e.g. ``"contiguous"``).
    """
    return METHOD_TO_OP_MAP.get(method_name, f"Tensor.{method_name}")
