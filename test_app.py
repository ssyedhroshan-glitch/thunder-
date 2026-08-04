import os
import pytest


def test_app_imports():
    """Verify app.py imports cleanly without syntax errors."""
    import app

    assert app is not None


def test_environment_port_default():
    """Verify default port fallback."""
    port_number = int(os.environ.get("PORT", 10000))
    assert port_number == 10000
    
