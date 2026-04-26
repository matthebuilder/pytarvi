import os
import shutil
import tempfile
from typing import Optional
from git import Repo
from loguru import logger


class GitHubManager:
    """Handles cloning and managing GitHub repositories for architecture extraction."""

    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="opencode_github_")
        logger.debug(f"Temporary directory created at: {self.temp_dir}")

    def clone_repository(self, repo_url: str) -> Optional[str]:
        """
        Clones a public GitHub repository to the temporary directory.

        Args:
            repo_url: The URL of the GitHub repository.

        Returns:
            The path to the cloned repository, or None if cloning failed.
        """
        try:
            logger.info(f"Cloning repository: {repo_url}")
            Repo.clone_from(repo_url, self.temp_dir)
            return self.temp_dir
        except Exception as e:
            logger.error(f"Failed to clone repository {repo_url}: {e}")
            return None

    def find_model_file(self) -> Optional[str]:
        """
        Searches for a Python file that likely contains a PyTorch model.
        Looks for files containing 'nn.Module'.

        Returns:
            The path to the most likely model file, or None if not found.
        """
        logger.info("Searching for model file containing 'nn.Module'...")
        for root, _, files in os.walk(self.temp_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            if "nn.Module" in content and "class" in content:
                                logger.info(f"Potential model file found: {file_path}")
                                return file_path
                    except Exception as e:
                        logger.warning(f"Could not read file {file_path}: {e}")

        logger.warning("No suitable model file found.")
        return None

    def cleanup(self) -> None:
        """Removes the temporary directory and its contents."""
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("Cleaned up temporary directory.")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
