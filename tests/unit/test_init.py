"""Test package initialization and imports."""

import dynamic_moe_router


def test_version():
    """Test that version is accessible."""
    assert hasattr(dynamic_moe_router, '__version__')
    assert isinstance(dynamic_moe_router.__version__, str)
    

def test_author():
    """Test that author is accessible."""
    assert hasattr(dynamic_moe_router, '__author__')
    assert isinstance(dynamic_moe_router.__author__, str)


def test_package_imports():
    """Test that package can be imported without errors."""
    # This is a basic smoke test to ensure the package structure is correct
    import dynamic_moe_router
    assert dynamic_moe_router is not None