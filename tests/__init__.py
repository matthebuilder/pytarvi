"""Tests for sources/__init__.py"""

import sources


def test_all_exports_importable() -> None:
    """Verify all __all__ symbols are importable."""
    for name in sources.__all__:
        assert hasattr(sources, name)
