"""Tests for sources.repo_handler — clone, find, import."""

from pathlib import Path

import pytest
import torch.nn as nn

from sources.exceptions import ModelImportError, RepoCloneError
from sources.repo_handler import (
    clone_repo,
    find_model_files,
    find_module_classes_ast,
    import_model,
    instantiate_model,
)


class TestCloneRepo:
    """Tests for clone_repo."""

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(RepoCloneError):
            clone_repo("https://github.com/nonexistent/repo-that-does-not-exist-12345")

    def test_empty_url_raises(self) -> None:
        with pytest.raises(RepoCloneError):
            clone_repo("")


class TestFindModelFiles:
    """Tests for find_model_files."""

    def test_finds_model_file(self, tmp_repo: Path) -> None:
        files = find_model_files(str(tmp_repo))
        assert len(files) >= 1
        assert any(f.stem == "model" for f in files)

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        files = find_model_files(str(tmp_path))
        assert files == []

    def test_ignores_test_dirs(self, tmp_path: Path) -> None:
        test_dir = tmp_path / "tests"
        test_dir.mkdir()
        (test_dir / "test_model.py").write_text(
            "import torch.nn as nn\nclass M(nn.Module): pass\n"
        )
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "model.py").write_text(
            "import torch.nn as nn\nclass Net(nn.Module): pass\n"
        )
        files = find_model_files(str(tmp_path))
        stems = [f.stem for f in files]
        assert "test_model" not in stems
        assert "model" in stems


class TestFindModuleClassesAst:
    """Tests for find_module_classes_ast."""

    def test_finds_class(self, tmp_repo: Path) -> None:
        model_file = tmp_repo / "model.py"
        classes = find_module_classes_ast(model_file)
        names = [name for name, _ in classes]
        assert "SimpleNet" in names

    def test_non_module_file(self, tmp_path: Path) -> None:
        py_file = tmp_path / "utils.py"
        py_file.write_text("def helper(): pass\n")
        classes = find_module_classes_ast(py_file)
        assert classes == []


class TestImportModel:
    """Tests for import_model."""

    def test_imports_model(self, tmp_repo: Path) -> None:
        model_file = tmp_repo / "model.py"
        classes = import_model(model_file, str(tmp_repo))
        assert len(classes) >= 1
        assert any(cls.__name__ == "SimpleNet" for cls in classes)

    def test_imported_classes_are_nn_module(self, tmp_repo: Path) -> None:
        model_file = tmp_repo / "model.py"
        classes = import_model(model_file, str(tmp_repo))
        for cls in classes:
            assert issubclass(cls, nn.Module)

    def test_syntax_error_raises(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("def broken(:\n")
        with pytest.raises(ModelImportError):
            import_model(bad_file, str(tmp_path))


class TestInstantiateModel:
    """Tests for instantiate_model."""

    def test_simple_model(self, tmp_repo: Path) -> None:
        model_file = tmp_repo / "model.py"
        classes = import_model(model_file, str(tmp_repo))
        simple_net = next(cls for cls in classes if cls.__name__ == "SimpleNet")
        model = instantiate_model(simple_net)
        assert isinstance(model, nn.Module)
