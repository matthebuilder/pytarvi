"""PyTorch Architecture Viewer -- Streamlit application entry point."""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import streamlit as st  # noqa: E402
import torch.nn as nn  # noqa: E402
from loguru import logger  # noqa: E402

from sources.config import COMMON_INPUT_SHAPES  # noqa: E402
from sources.exceptions import (  # noqa: E402
    AppError,
    ModelImportError,
    NoModelError,
    RepoCloneError,
)
from sources.graph_extractor import extract_architecture  # noqa: E402
from sources.graph_model import ArchitectureGraph  # noqa: E402
from sources.node_editor_renderer import render_graph  # noqa: E402
from sources.repo_handler import (  # noqa: E402
    clone_repo,
    find_model_files,
    import_model,
    instantiate_model,
)

st.set_page_config(
    page_title="PyTorch Architecture Viewer",
    layout="wide",
)

st.title("PyTorch Architecture Viewer")
st.caption(
    "Enter a GitHub repository URL to extract and visualize the PyTorch "
    "model architecture as an interactive node graph."
)


def _clone_and_scan(url: str) -> Tuple[str, List[Tuple[Type[nn.Module], str]]]:
    """Clone a repo and scan for nn.Module subclasses.

    Args:
        url: GitHub repository URL.

    Returns:
        Tuple of ``(repo_dir, model_classes)`` where each entry is
        ``(class_type, file_path_str)``.

    Raises:
        RepoCloneError: If cloning fails.
        NoModelError: If no model classes are found.
    """
    repo_dir = clone_repo(url)
    model_files = find_model_files(repo_dir)

    if not model_files:
        raise NoModelError(f"No model files found in {url}")

    all_classes: List[Tuple[Type[nn.Module], str]] = []
    for file_path in model_files:
        classes = import_model(file_path, repo_dir)
        for cls in classes:
            all_classes.append((cls, str(file_path)))

    if not all_classes:
        raise NoModelError(f"No nn.Module subclasses found in {url}")

    return repo_dir, all_classes


def _get_input_shape(
    model_class: Type[nn.Module],
) -> Tuple[int, ...]:
    """Determine a reasonable input shape for a model class.

    Args:
        model_class: The model class to infer input shape for.

    Returns:
        A default input shape tuple.
    """
    name_lower = model_class.__name__.lower()

    vision_kws = ("conv", "vision", "image", "resnet", "vgg", "effici", "mobilenet")
    if any(kw in name_lower for kw in vision_kws):
        return COMMON_INPUT_SHAPES["vision_cnn"]
    nlp_kws = ("transformer", "bert", "gpt", "llm", "attention")
    if any(kw in name_lower for kw in nlp_kws):
        return COMMON_INPUT_SHAPES["nlp_transformer"]
    if any(kw in name_lower for kw in ("lstm", "gru", "rnn", "seq")):
        return COMMON_INPUT_SHAPES["nlp_rnn"]

    return COMMON_INPUT_SHAPES["vision_cnn_small"]


def _display_summary(graph: ArchitectureGraph) -> None:
    """Display an architecture summary table.

    Args:
        graph: The :class:`ArchitectureGraph` to summarize.
    """
    st.subheader("Architecture Summary")

    rows: List[Dict[str, str]] = []
    for node in graph.nodes:
        rows.append(
            {
                "Layer": node.name,
                "Type": node.op_type,
                "Category": node.category,
                "Input Shape": str(list(node.input_shape)) if node.input_shape else "-",
                "Output Shape": str(list(node.output_shape))
                if node.output_shape
                else "-",
                "Params": (
                    ", ".join(f"{k}={v}" for k, v in node.params.items())
                    if node.params
                    else "-"
                ),
            }
        )

    st.dataframe(rows, use_container_width=True)


with st.sidebar:
    repo_url = st.text_input(
        "GitHub Repository URL",
        placeholder="https://github.com/user/repo",
    )

    shape_preset = st.selectbox(
        "Input Shape Preset",
        list(COMMON_INPUT_SHAPES.keys()),
        index=0,
    )

    custom_shape = st.text_input(
        "Custom Input Shape (comma-separated)",
        value="",
        help="Override the preset, e.g. 1,3,224,224",
    )

    analyze_button = st.button("Analyze", type="primary")

if analyze_button and repo_url:
    with st.spinner("Cloning repository and scanning for models..."):
        try:
            repo_dir, model_classes = _clone_and_scan(repo_url)
            st.session_state["repo_dir"] = repo_dir
            st.session_state["model_classes"] = [(cls, fp) for cls, fp in model_classes]
            st.session_state["source_repo"] = repo_url
        except RepoCloneError as exc:
            st.error(f"Failed to clone repository: {exc}")
            st.session_state.pop("model_classes", None)
        except NoModelError as exc:
            st.warning(str(exc))
            st.session_state.pop("model_classes", None)
        except AppError as exc:
            st.error(str(exc))
            st.session_state.pop("model_classes", None)

if "model_classes" in st.session_state:
    class_names = [cls.__name__ for cls, _ in st.session_state["model_classes"]]

    with st.sidebar:
        selected_name = st.selectbox(
            "Select Model Class",
            class_names,
            index=0,
        )

    selected_class: Optional[Type[nn.Module]] = None
    selected_file: Optional[str] = None
    for cls, fp in st.session_state["model_classes"]:
        if cls.__name__ == selected_name:
            selected_class = cls
            selected_file = fp
            break

    if selected_class is not None:
        input_shape = COMMON_INPUT_SHAPES[shape_preset]
        if custom_shape.strip():
            try:
                input_shape = tuple(
                    int(x.strip()) for x in custom_shape.split(",") if x.strip()
                )
            except ValueError:
                st.warning("Invalid custom shape format. Using preset.")

        with st.sidebar:
            st.info(f"Input shape: {input_shape}")

        if st.sidebar.button("Extract Architecture", type="primary"):
            with st.spinner("Extracting architecture..."):
                try:
                    model = instantiate_model(selected_class)
                    from pathlib import Path

                    graph = extract_architecture(
                        model=model,
                        input_shape=input_shape,
                        file_path=Path(selected_file) if selected_file else None,
                        model_name=selected_class.__name__,
                        source_repo=st.session_state.get("source_repo", ""),
                    )
                    st.session_state["arch_graph"] = graph
                except ModelImportError as exc:
                    st.error(f"Failed to instantiate model: {exc}")
                except NoModelError as exc:
                    st.error(str(exc))
                except Exception as exc:
                    st.error(f"Extraction failed: {exc}")
                    logger.exception("Architecture extraction failed")

if "arch_graph" in st.session_state:
    graph = st.session_state["arch_graph"]

    st.subheader(f"{graph.model_name} Architecture")
    result = render_graph(graph)

    with st.expander("Architecture Summary"):
        _display_summary(graph)

    with st.expander("Graph JSON"):
        if result is not None:
            st.json(result)
        else:
            st.info("Interact with the graph above to see the JSON state.")
