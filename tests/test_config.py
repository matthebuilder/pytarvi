"""Tests for sources.config — mapping completeness and helpers."""


import torch
import torch.nn.functional as F  # noqa: N812

from sources.config import (
    CATEGORY_MAP,
    FUNCTIONAL_TO_MODULE_MAP,
    NODE_COLORS,
    get_category,
    get_color,
    get_op_type_for_function,
    get_op_type_for_method,
)


class TestNodeColors:
    """Tests for NODE_COLORS mapping."""

    def test_all_categories_have_colors(self) -> None:
        for _op_type, category in CATEGORY_MAP.items():
            assert category in NODE_COLORS, f"Missing color for category: {category}"

    def test_default_category_has_color(self) -> None:
        assert "Other" in NODE_COLORS


class TestCategoryMap:
    """Tests for CATEGORY_MAP mapping."""

    def test_covers_common_layers(self) -> None:
        common = [
            "Conv2d", "Linear", "BatchNorm2d", "ReLU",
            "Dropout", "MaxPool2d", "LSTM",
        ]
        for layer in common:
            assert layer in CATEGORY_MAP, f"Missing category for: {layer}"

    def test_categories_are_valid(self) -> None:
        for _op_type, category in CATEGORY_MAP.items():
            assert category in NODE_COLORS, f"Category {category} has no color"


class TestFunctionalToModuleMap:
    """Tests for FUNCTIONAL_TO_MODULE_MAP."""

    def test_covers_common_functionals(self) -> None:
        assert F.relu in FUNCTIONAL_TO_MODULE_MAP
        assert F.dropout in FUNCTIONAL_TO_MODULE_MAP
        assert F.conv2d in FUNCTIONAL_TO_MODULE_MAP
        assert F.linear in FUNCTIONAL_TO_MODULE_MAP

    def test_torch_top_level(self) -> None:
        assert torch.relu in FUNCTIONAL_TO_MODULE_MAP
        assert torch.flatten in FUNCTIONAL_TO_MODULE_MAP
        assert torch.cat in FUNCTIONAL_TO_MODULE_MAP

    def test_operator_mappings(self) -> None:
        import operator
        assert operator.add in FUNCTIONAL_TO_MODULE_MAP


class TestGetCategory:
    """Tests for get_category helper."""

    def test_known_type(self) -> None:
        assert get_category("Conv2d") == "Convolution"

    def test_unknown_type_returns_default(self) -> None:
        assert get_category("UnknownLayer") == "Other"


class TestGetColor:
    """Tests for get_color helper."""

    def test_known_category(self) -> None:
        assert get_color("Convolution") == "#3b82f6"

    def test_unknown_category_returns_gray(self) -> None:
        assert get_color("NonExistent") == "#6b7280"


class TestGetOpTypeForFunction:
    """Tests for get_op_type_for_function helper."""

    def test_known_function(self) -> None:
        assert get_op_type_for_function(F.relu) == "ReLU"

    def test_unknown_function_returns_name(self) -> None:
        def custom_op(x: int) -> int:
            return x
        assert get_op_type_for_function(custom_op) == "custom_op"


class TestGetOpTypeForMethod:
    """Tests for get_op_type_for_method helper."""

    def test_view(self) -> None:
        assert get_op_type_for_method("view") == "View"

    def test_contiguous_skipped(self) -> None:
        assert get_op_type_for_method("contiguous") == ""

    def test_unknown_method(self) -> None:
        result = get_op_type_for_method("custom_method")
        assert "Tensor." in result
