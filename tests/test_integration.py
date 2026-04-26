import os
import pytest
from unittest.mock import patch, MagicMock
from sources.github_manager import GitHubManager
from sources.extractor import ArchitectureExtractor
from sources.mapper import GraphMapper


class TestIntegration:
    @patch("sources.github_manager.Repo.clone_from")
    def test_full_pipeline_flow(self, mock_clone):
        # Setup: Mock the GitHubManager to return a dummy file path
        mock_clone.return_value = MagicMock()

        repo_url = "https://github.com/user/repo"

        with GitHubManager() as gh:
            # 1. Clone
            path = gh.clone_repository(repo_url)

            # Create a dummy model file in the cloned path
            model_file = os.path.join(path, "model.py")
            with open(model_file, "w") as f:
                f.write("""
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(10, 10)
        self.l2 = nn.Linear(10, 5)
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x
                """)

            # 2. Find file
            found_file = gh.find_model_file()
            assert found_file == model_file

            # 3. Extract
            extractor = ArchitectureExtractor(found_file)
            nodes, edges = extractor.extract()
            assert len(nodes) == 2
            assert len(edges) == 1

            # 4. Map
            mapper = GraphMapper()
            node_defs, initial_nodes, initial_connections = mapper.map_to_editor(
                nodes, edges
            )
            assert len(initial_nodes) == 2
            assert len(initial_connections) == 1
            assert initial_nodes[0]["id"] == "l1"
