"""Shared pytest fixtures for the test suite."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from sources.graph_model import ArchitectureGraph


class SimpleCNN(nn.Module):
    """Minimal CNN with Conv2d -> BN -> ReLU -> Pool -> Linear -> ReLU -> Linear."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(16, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class ResidualBlock(nn.Module):
    """BasicBlock with skip connection for testing branching."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out


class FunctionalModel(nn.Module):
    """Model using F.relu, F.dropout, F.interpolate for functional detection."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class SequentialModel(nn.Module):
    """Model using nn.Sequential container."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(16, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@pytest.fixture
def simple_cnn() -> SimpleCNN:
    """Provide a SimpleCNN model instance."""
    return SimpleCNN()


@pytest.fixture
def residual_block() -> ResidualBlock:
    """Provide a ResidualBlock model instance."""
    return ResidualBlock()


@pytest.fixture
def functional_model() -> FunctionalModel:
    """Provide a FunctionalModel instance."""
    return FunctionalModel()


@pytest.fixture
def sequential_model() -> SequentialModel:
    """Provide a SequentialModel instance."""
    return SequentialModel()


@pytest.fixture
def sample_arch_graph() -> "ArchitectureGraph":
    """Provide a sample ArchitectureGraph with known structure."""
    from sources.graph_model import ArchEdge, ArchitectureGraph, ArchNode

    nodes = [
        ArchNode(
            id="conv1", name="conv1", op_type="Conv2d",
            category="Convolution", color="#3b82f6",
        ),
        ArchNode(
            id="relu1", name="relu1", op_type="ReLU",
            category="Activation", color="#4ade80",
        ),
        ArchNode(
            id="fc1", name="fc1", op_type="Linear",
            category="Linear", color="#60a5fa",
        ),
    ]
    edges = [
        ArchEdge(id="e0", source_id="conv1", target_id="relu1"),
        ArchEdge(id="e1", source_id="relu1", target_id="fc1"),
    ]
    return ArchitectureGraph(
        nodes=nodes,
        edges=edges,
        model_name="TestModel",
        source_repo="https://github.com/test/repo",
        input_shape=(1, 3, 224, 224),
    )


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Create a temporary directory with a sample model.py file."""
    model_code = '''
import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 13 * 13, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
'''
    model_file = tmp_path / "model.py"
    model_file.write_text(model_code)
    return tmp_path
