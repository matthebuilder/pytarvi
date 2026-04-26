import pytest
from unittest.mock import patch, MagicMock
import os
import tempfile
from sources.github_manager import GitHubManager

class TestGitHubManager:
    @patch("sources.github_manager.Repo.clone_from")
    def test_clone_repository_success(self, mock_clone):
        mock_clone.return_value = MagicMock()
        
        with GitHubManager() as gh:
            url = "https://github.com/user/repo"
            path = gh.clone_repository(url)
            
            assert path is not None
            assert os.path.exists(path)
            mock_clone.assert_called_once_with(url, gh.temp_dir)

    @patch("sources.github_manager.Repo.clone_from")
    def test_clone_repository_failure(self, mock_clone):
        mock_clone.side_effect = Exception("Clone failed")
        
        with GitHubManager() as gh:
            url = "https://github.com/invalid/repo"
            path = gh.clone_repository(url)
            
            assert path is None

    def test_find_model_file_success(self):
        with GitHubManager() as gh:
            # Create a dummy model file in temp dir
            model_path = os.path.join(gh.temp_dir, "model.py")
            with open(model_path, "w") as f:
                f.write("import torch.nn as nn\nclass MyModel(nn.Module): pass")
            
            found_path = gh.find_model_file()
            assert found_path == model_path

    def test_find_model_file_not_found(self):
        with GitHubManager() as gh:
            # Create a non-model file
            with open(os.path.join(gh.temp_dir, "readme.txt"), "w") as f:
                f.write("Just a text file")
            
            found_path = gh.find_model_file()
            assert found_path is None

    def test_cleanup(self):
        gh = GitHubManager()
        temp_path = gh.temp_dir
        assert os.path.exists(temp_path)
        
        gh.cleanup()
        assert not os.path.exists(temp_path)
