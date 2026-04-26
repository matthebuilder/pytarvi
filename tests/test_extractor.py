import os
import tempfile
from typing import List, Tuple
from unittest.mock import patch, MagicMock
from sources.extractor import ArchitectureExtractor
from sources.models import Node, Edge

def create_tmp_model(code: str) -> str:
    """Helper to create a temporary python file with given code."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as tmp:
        tmp.write(code)
        return tmp.name

class TestArchitectureExtractor:
    def test_linear_flow(self):
        code = """
import torch.nn as nn
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.fc(x)
        return x
        """
        path = create_tmp_model(code)
        try:
            extractor = ArchitectureExtractor(path)
            nodes, edges = extractor.extract()
            
            node_names = {n.name for n in nodes}
            assert {"conv1", "relu", "fc"}.issubset(node_names)
            assert len(nodes) == 3
            assert len(edges) == 2
        finally:
            os.remove(path)

    def test_sequential_flow(self):
        code = """
import torch.nn as nn
class SeqModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU()
        )
        self.classifier = nn.Linear(16, 10)
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
        """
        path = create_tmp_model(code)
        try:
            extractor = ArchitectureExtractor(path)
            nodes, edges = extractor.extract()
            
            node_names = {n.name for n in nodes}
            assert {"features", "classifier"}.issubset(node_names)
            assert len(nodes) == 2
            assert len(edges) == 1
        finally:
            os.remove(path)

    def test_residual_connection(self):
        code = """
import torch.nn as nn
class ResModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 16, 3)
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out) + out
        return out
        """
        path = create_tmp_model(code)
        try:
            extractor = ArchitectureExtractor(path)
            nodes, edges = extractor.extract()
            
            node_names = {n.name for n in nodes}
            assert {"conv1", "conv2"}.issubset(node_names)
            assert len(edges) >= 1
        finally:
            os.remove(path)

    def test_conditional_logic(self):
        code = """
import torch.nn as nn
class CondModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)
    def forward(self, x):
        if True:
            x = self.layer1(x)
        else:
            x = self.layer2(x)
        return x
        """
        path = create_tmp_model(code)
        try:
            extractor = ArchitectureExtractor(path)
            nodes, edges = extractor.extract()
            
            node_names = {n.name for n in nodes}
            assert {"layer1", "layer2"}.issubset(node_names)
        finally:
            os.remove(path)

    def test_no_model_class(self):
        code = "print('Hello')"
        path = create_tmp_model(code)
        try:
            extractor = ArchitectureExtractor(path)
            nodes, edges = extractor.extract()
            assert nodes == []
            assert edges == []
        finally:
            os.remove(path)
