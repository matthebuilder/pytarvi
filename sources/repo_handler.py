"""GitHub repository handling: clone, find model files, dynamic import."""

import ast
import importlib.util
import inspect
import subprocess
import sys
import tempfile
import types
import unittest.mock
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Type

import torch.nn as nn
from loguru import logger

from sources.exceptions import ModelImportError, RepoCloneError


def clone_repo(url: str, target_dir: Optional[str] = None) -> str:
    """Shallow-clone a GitHub repository.

    Args:
        url: GitHub repository URL (HTTPS or SSH).
        target_dir: Local directory to clone into. If ``None``, a
            temporary directory is created.

    Returns:
        Absolute path to the cloned repository directory.

    Raises:
        RepoCloneError: If the git clone command fails.
    """
    if target_dir is None:
        target_dir = tempfile.mkdtemp(prefix="pytorch_arch_")

    logger.info(f"Cloning {url} into {target_dir}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, target_dir],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.CalledProcessError as exc:
        raise RepoCloneError(
            f"Failed to clone {url}: {exc.stderr.strip()}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RepoCloneError(
            f"Clone of {url} timed out after 120s"
        ) from exc

    return target_dir


def _has_module_subclass(tree: ast.AST) -> bool:
    """Check if an AST tree contains a class that subclasses nn.Module.

    Args:
        tree: Parsed Python AST.

    Returns:
        ``True`` if any class definition subclasses ``nn.Module``.
    """
    module_names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "nn" in node.module:
                for alias in node.names:
                    if alias.name == "Module":
                        module_names.add(
                            alias.asname if alias.asname else "Module"
                        )
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch.nn":
                    module_names.add("nn")
                if alias.name == "nn":
                    module_names.add("nn")

    if not module_names:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = _get_base_name(base)
            if base_name and (
                base_name == "Module"
                or base_name.endswith(".Module")
                or (base_name.startswith("nn.") and base_name.endswith("Module"))
            ):
                return True
    return False


def _get_base_name(base: ast.expr) -> Optional[str]:
    """Extract a string name from an AST base class expression.

    Args:
        base: An AST expression from a class's base class list.

    Returns:
        String representation of the base, or ``None``.
    """
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        value_name = _get_base_name(base.value)
        if value_name is not None:
            return f"{value_name}.{base.attr}"
        return base.attr
    return None


def find_model_files(repo_dir: str) -> List[Path]:
    """Find Python files that likely define nn.Module subclasses.

    Uses AST analysis to detect ``nn.Module`` subclass definitions.
    Also checks common naming conventions for model files.

    Args:
        repo_dir: Path to the cloned repository.

    Returns:
        Sorted, deduplicated list of paths to Python files.
    """
    repo_path = Path(repo_dir)
    found: Set[Path] = set()

    common_names: Set[str] = {
        "model", "models", "network", "networks", "arch", "architecture",
        "net", "module",
    }

    for py_file in repo_path.rglob("*.py"):
        rel = py_file.relative_to(repo_path)
        parts = rel.parts

        if any(part.startswith(".") for part in parts):
            continue
        if any(part in ("test", "tests", "__pycache__", ".git") for part in parts):
            continue

        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            if _has_module_subclass(tree):
                found.add(py_file)
                continue
        except (SyntaxError, UnicodeDecodeError):
            pass

        stem = py_file.stem.lower()
        if stem in common_names or stem.startswith("model_"):
            found.add(py_file)

        for part in parts:
            if part.lower().rstrip("s") in common_names:
                found.add(py_file)
                break

    return sorted(found)


def find_module_classes_ast(file_path: Path) -> List[Tuple[str, Path]]:
    """Find nn.Module subclass names in a file using AST analysis.

    Does not import the file -- purely static analysis.

    Args:
        file_path: Path to the Python file.

    Returns:
        List of ``(class_name, file_path)`` tuples.
    """
    try:
        tree = ast.parse(file_path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as exc:
        logger.warning(f"Cannot parse {file_path}: {exc}")
        return []

    results: List[Tuple[str, Path]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = _get_base_name(base)
            if base_name and (
                base_name == "Module"
                or base_name.endswith(".Module")
                or (base_name.startswith("nn.") and base_name.endswith("Module"))
            ):
                results.append((node.name, file_path))
                break

    return results


def import_model(
    file_path: Path, repo_dir: str
) -> List[Type[nn.Module]]:
    """Dynamically import a Python file and find nn.Module subclasses.

    Adds the repo root to ``sys.path`` so relative imports work. Missing
    dependencies are mocked with :class:`unittest.mock.MagicMock`.

    Args:
        file_path: Path to the Python file containing model definitions.
        repo_dir: Root directory of the cloned repository.

    Returns:
        List of instantiable ``nn.Module`` subclass types.

    Raises:
        ModelImportError: If the file cannot be imported due to syntax
            errors or other unrecoverable failures.
    """
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    module_name = f"dynamic_model_{file_path.stem}"

    original_modules: Dict[str, Optional[types.ModuleType]] = {}
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise ModelImportError(f"Cannot create import spec for {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        except ImportError as exc:
            missing_name = _extract_missing_module_name(exc)
            if missing_name:
                logger.info(
                    f"Mocking missing dependency: {missing_name}"
                )
                original_modules[missing_name] = sys.modules.get(
                    missing_name, None
                )
                sys.modules[missing_name] = unittest.mock.MagicMock()
                try:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)  # type: ignore[union-attr]
                except ImportError:
                    pass
            else:
                logger.warning(f"Import error in {file_path}: {exc}")

    except SyntaxError as exc:
        raise ModelImportError(
            f"Syntax error in {file_path}: {exc}"
        ) from exc
    except Exception as exc:
        logger.warning(f"Failed to import {file_path}: {exc}")
        return []
    finally:
        for mod_name, original in original_modules.items():
            if original is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = original

    model_classes: List[Type[nn.Module]] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, nn.Module)
            and obj is not nn.Module
        ):
            model_classes.append(obj)

    return model_classes


def _extract_missing_module_name(exc: ImportError) -> Optional[str]:
    """Extract the missing module name from an ImportError.

    Args:
        exc: The :class:`ImportError` to parse.

    Returns:
        The missing module name, or ``None`` if it cannot be determined.
    """
    if exc.name:
        return exc.name
    msg = str(exc)
    if "No module named" in msg:
        parts = msg.split("'")
        if len(parts) >= 2:
            return parts[1]
    return None


def instantiate_model(model_class: Type[nn.Module]) -> nn.Module:
    """Instantiate an nn.Module class with minimal default arguments.

    Inspects the ``__init__`` signature and fills in defaults where
    possible. Parameters without defaults are set to reasonable
    placeholder values.

    Args:
        model_class: The ``nn.Module`` subclass to instantiate.

    Returns:
        An instance of the model in eval mode.

    Raises:
        ModelImportError: If the model cannot be instantiated.
    """
    try:
        return model_class()
    except TypeError:
        pass

    try:
        sig = inspect.signature(model_class.__init__)
        kwargs: Dict[str, object] = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            if param.default is not inspect.Parameter.empty:
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            kwargs[param_name] = _default_for_param(param_name)

        return model_class(**kwargs)
    except Exception as exc:
        raise ModelImportError(
            f"Cannot instantiate {model_class.__name__}: {exc}"
        ) from exc


def _default_for_param(name: str) -> object:
    """Return a sensible default value for a constructor parameter.

    Args:
        name: Parameter name (used as a heuristic).

    Returns:
        A default value guess.
    """
    name_lower = name.lower()
    if any(kw in name_lower for kw in ("channel", "dim", "feature", "hidden")):
        return 64
    if any(kw in name_lower for kw in ("num_class", "n_class", "output_dim")):
        return 10
    if any(kw in name_lower for kw in ("layer", "block", "depth", "num")):
        return 3
    if any(kw in name_lower for kw in ("size", "length", "seq")):
        return 128
    if "rate" in name_lower or "dropout" in name_lower:
        return 0.1
    if "head" in name_lower:
        return 4
    return 1
