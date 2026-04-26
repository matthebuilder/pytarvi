# Building Full Documentation

This project uses **Sphinx** to generate comprehensive technical documentation, including API references derived from source code docstrings.

## 🛠️ Documentation Tooling
To build the documentation, you need the following installed:
- `sphinx`
- `sphinx-rtd-theme`
- `napoleon` (included with Sphinx)

These are listed in the `requirements.txt` file.

## 🏗️ Build Process

### 1. Generate HTML Documentation
Run the following command from the project root:
```bash
sphinx-build -b html doc/source doc/build
```
The generated documentation will be available in the `doc/build` folder. Open `index.html` in any browser to view it.

### 2. Cleaning Previous Builds
To remove old documentation files and start fresh:
```bash
rm -rf doc/build
```

## ✍️ Contributing to Documentation

### Writing Docstrings
We use the **Google Style** for docstrings. Please ensure all public classes and methods follow this format:

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Brief description of the function.

    Args:
        param1: Description of the first parameter.
        param2: Description of the second parameter.

    Returns:
        Description of the return value.

    Raises:
        ValueError: If param1 is negative.
    """
    return True
```

### Adding New Pages
1. Create a new `.rst` file in `doc/source/`.
2. Add the new file to the `toctree` in `doc/source/index.rst`.
3. Re-run the build command.

[← Back to Home](../README.md)
