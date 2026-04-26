"""Tests for sources.graph_extractor — FX trace, hooks, AST, orchestrator."""

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from sources.exceptions import NoModelError
from sources.graph_extractor import (
    _extract_layer_params,
    extract_architecture,
    extract_via_ast,
    extract_via_fx,
    extract_via_hooks,
)


class TestExtractViaFx:
    """Tests for FX symbolic trace extraction."""

    def test_simple_cnn(self, simple_cnn: nn.Module) -> None:
        graph = extract_via_fx(simple_cnn, (1, 3, 32, 32), "SimpleCNN")
        assert len(graph.nodes) > 0
        assert len(graph.edges) > 0
        op_types = {n.op_type for n in graph.nodes}
        assert "Conv2d" in op_types
        assert "Linear" in op_types

    def test_residual_block_has_add(self, residual_block: nn.Module) -> None:
        graph = extract_via_fx(residual_block, (1, 3, 8, 8), "ResidualBlock")
        op_types = {n.op_type for n in graph.nodes}
        assert "Add" in op_types

    def test_residual_block_branching(self, residual_block: nn.Module) -> None:
        graph = extract_via_fx(residual_block, (1, 3, 8, 8), "ResidualBlock")
        add_node = next(n for n in graph.nodes if n.op_type == "Add")
        incoming = [e for e in graph.edges if e.target_id == add_node.id]
        assert len(incoming) == 2

    def test_functional_ops_detected(self, functional_model: nn.Module) -> None:
        graph = extract_via_fx(functional_model, (1, 3, 16, 16), "FunctionalModel")
        op_types = {n.op_type for n in graph.nodes}
        assert "ReLU" in op_types
        assert "Dropout" in op_types

    def test_shapes_populated(self, simple_cnn: nn.Module) -> None:
        graph = extract_via_fx(simple_cnn, (1, 3, 32, 32), "SimpleCNN")
        nodes_with_shapes = [n for n in graph.nodes if n.output_shape is not None]
        assert len(nodes_with_shapes) > 0

    def test_model_name_set(self, simple_cnn: nn.Module) -> None:
        graph = extract_via_fx(simple_cnn, (1, 3, 32, 32), "MyModel")
        assert graph.model_name == "MyModel"

    def test_input_shape_recorded(self, simple_cnn: nn.Module) -> None:
        graph = extract_via_fx(simple_cnn, (1, 3, 32, 32), "SimpleCNN")
        assert graph.input_shape == (1, 3, 32, 32)


class TestExtractViaHooks:
    """Tests for hook-based extraction."""

    def test_simple_cnn(self, simple_cnn: nn.Module) -> None:
        graph = extract_via_hooks(simple_cnn, (1, 3, 32, 32), "SimpleCNN")
        assert len(graph.nodes) > 0
        op_types = {n.op_type for n in graph.nodes}
        assert "Conv2d" in op_types

    def test_shapes_populated(self, simple_cnn: nn.Module) -> None:
        simple_cnn.eval()
        graph = extract_via_hooks(simple_cnn, (1, 3, 32, 32), "SimpleCNN")
        nodes_with_input = [n for n in graph.nodes if n.input_shape is not None]
        assert len(nodes_with_input) > 0

    def test_sequential_edges(self, simple_cnn: nn.Module) -> None:
        graph = extract_via_hooks(simple_cnn, (1, 3, 32, 32), "SimpleCNN")
        assert len(graph.edges) == len(graph.nodes) - 1

    def test_no_functional_ops(self, functional_model: nn.Module) -> None:
        functional_model.eval()
        graph = extract_via_hooks(functional_model, (1, 3, 16, 16), "FunctionalModel")
        op_types = {n.op_type for n in graph.nodes}
        assert "ReLU" not in op_types or "ReLU" in op_types


class TestExtractViaAst:
    """Tests for AST-based extraction."""

    def test_finds_layers(self, tmp_repo: Path) -> None:
        model_file = tmp_repo / "model.py"
        graph = extract_via_ast(model_file, "SimpleNet")
        op_types = {n.op_type for n in graph.nodes}
        assert "Conv2d" in op_types
        assert "ReLU" in op_types
        assert "Linear" in op_types

    def test_linear_chain_edges(self, tmp_repo: Path) -> None:
        model_file = tmp_repo / "model.py"
        graph = extract_via_ast(model_file, "SimpleNet")
        assert len(graph.edges) == len(graph.nodes) - 1

    def test_no_model_raises(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.py"
        empty_file.write_text("x = 1\n")
        with pytest.raises(NoModelError):
            extract_via_ast(empty_file, "MissingModel")


class TestExtractArchitecture:
    """Tests for the orchestrator function."""

    def test_fx_preferred(self, simple_cnn: nn.Module) -> None:
        graph = extract_architecture(simple_cnn, (1, 3, 32, 32), model_name="SimpleCNN")
        assert len(graph.nodes) > 0

    def test_fallback_to_hooks(self) -> None:
        class UntraceableModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(10, 5)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if x.sum().item() > 0:
                    return self.fc(x)
                return x

        model = UntraceableModel()
        graph = extract_architecture(model, (1, 10), model_name="UntraceableModel")
        assert len(graph.nodes) > 0

    def test_fallback_to_ast(self, tmp_repo: Path) -> None:
        model_file = tmp_repo / "model.py"
        graph = extract_architecture(
            model=None,  # type: ignore[arg-type]
            input_shape=(1, 3, 32, 32),
            file_path=model_file,
            model_name="SimpleNet",
        )
        assert len(graph.nodes) > 0

    def test_no_model_raises(self) -> None:
        with pytest.raises(NoModelError):
            extract_architecture(
                model=None,  # type: ignore[arg-type]
                input_shape=(1, 10),
                model_name="Nothing",
            )


class TestExtractLayerParams:
    """Tests for _extract_layer_params helper."""

    def test_conv2d(self) -> None:
        module = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        params = _extract_layer_params(module)
        assert params["in_channels"] == 3
        assert params["out_channels"] == 16
        assert params["kernel_size"] == (3, 3)
        assert params["stride"] == (2, 2)

    def test_linear(self) -> None:
        module = nn.Linear(64, 10)
        params = _extract_layer_params(module)
        assert params["in_features"] == 64
        assert params["out_features"] == 10

    def test_batchnorm(self) -> None:
        module = nn.BatchNorm2d(16)
        params = _extract_layer_params(module)
        assert params["num_features"] == 16

    def test_dropout(self) -> None:
        module = nn.Dropout(p=0.3)
        params = _extract_layer_params(module)
        assert params["p"] == 0.3

    def test_lstm(self) -> None:
        module = nn.LSTM(128, 256, num_layers=2)
        params = _extract_layer_params(module)
        assert params["input_size"] == 128
        assert params["hidden_size"] == 256
        assert params["num_layers"] == 2

    def test_unknown_module(self) -> None:
        class CustomLayer(nn.Module):
            pass

        params = _extract_layer_params(CustomLayer())
        assert params == {}
