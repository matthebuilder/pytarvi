import streamlit as st
from loguru import logger
from sources.utils.logging_config import setup_logging
from sources.github_manager import GitHubManager
from sources.extractor import ArchitectureExtractor
from sources.mapper import GraphMapper
from streamlit_node_editor import st_node_editor

# Initialize logging
setup_logging()


st.set_page_config(page_title="PyTorch Architecture Visualizer", layout="wide")


def main():
    st.title("PyTorch Architecture Visualizer")
    st.markdown("""
    Extract and visualize the architecture of a PyTorch model from a GitHub repository.
    """)

    with st.sidebar:
        st.header("Configuration")
        repo_url = st.text_input(
            "GitHub Repository URL", placeholder="https://github.com/username/repo"
        )
        analyze_btn = st.button("Analyze Architecture", type="primary")

        st.divider()
        st.info(
            "This tool uses static AST analysis to extract the model architecture without executing the code."
        )

    if analyze_btn:
        if not repo_url:
            st.error("Please provide a GitHub repository URL.")
            return

        with st.spinner("Analyzing repository..."):
            try:
                with GitHubManager() as gh:
                    repo_path = gh.clone_repository(repo_url)
                    if not repo_path:
                        st.error(
                            "Failed to clone the repository. Please check the URL."
                        )
                        return

                    model_file = gh.find_model_file()
                    if not model_file:
                        st.error(
                            "Could not find a PyTorch model file (inheriting from nn.Module) in the repository."
                        )
                        return

                    logger.info(f"Extracting architecture from {model_file}")
                    extractor = ArchitectureExtractor(model_file)
                    nodes, edges = extractor.extract()

                    if not nodes:
                        st.warning(
                            "No layers were extracted. The model might be too complex for static analysis."
                        )
                        return

                    mapper = GraphMapper()
                    node_defs, initial_nodes, initial_connections = (
                        mapper.map_to_editor(nodes, edges)
                    )

                    st.success(
                        f"Successfully extracted {len(nodes)} layers and {len(edges)} connections!"
                    )

                    # Visualization
                    st.subheader("Model Graph")

                    # Initialize the node editor
                    st_node_editor(
                        node_defs=node_defs,
                        initial_nodes=initial_nodes,
                        initial_connections=initial_connections,
                    )

                    # In a real streamlit-node-editor implementation, you might handle updates:
                    # updated_data = editor.get_data()

                    # Display details of extracted nodes in a table
                    with st.expander("View Layer Details"):
                        st.table(
                            [
                                {
                                    "Layer": n.name,
                                    "Type": n.type,
                                    "Params": str(n.params),
                                }
                                for n in nodes
                            ]
                        )

            except Exception as e:
                logger.exception(f"An unexpected error occurred: {e}")
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
