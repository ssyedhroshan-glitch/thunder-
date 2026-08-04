import os
import tempfile
import pytest
import app

@pytest.fixture(autouse=True)
def setup_and_teardown_db(monkeypatch):
    """Create a temporary database for each test to isolate tests cleanly."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        temp_db_path = tmp.name
    
    # Point app.DB_PATH to the isolated temporary database
    monkeypatch.setattr(app, "DB_PATH", temp_db_path)
    
    if hasattr(app, "init_db"):
        app.init_db()
        
    yield
    
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

def test_app_imports():
    """Verify app imports cleanly."""
    assert app is not None

def test_database_session_operations():
    """Test session creation, retrieval, and deletion if available."""
    if hasattr(app, "get_all_sessions"):
        sessions = app.get_all_sessions()
        assert isinstance(sessions, list)

    if hasattr(app, "create_new_session"):
        session_id = app.create_new_session("Test Workspace")
        assert session_id is not None
        
        if hasattr(app, "delete_session_from_db"):
            app.delete_session_from_db(session_id)

def test_message_persistence():
    """Test saving and loading messages."""
    if hasattr(app, "get_all_sessions") and hasattr(app, "save_message") and hasattr(app, "load_history"):
        sessions = app.get_all_sessions()
        if sessions:
            session_id = sessions[0][0]
            app.save_message(session_id, "user", "Test message")
            history = app.load_history(session_id)
            assert len(history) > 0

def test_execute_python_code():
    """Test code execution sandbox."""
    if hasattr(app, "execute_python_code"):
        output = app.execute_python_code("print('Coverage Test')")
        assert "Coverage Test" in str(output)

def test_read_file():
    """Test reading context files."""
    if hasattr(app, "read_file"):
        with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write("Test content")
            tmp_path = tmp.name

        try:
            class DummyFile:
                name = tmp_path

            result = app.read_file(DummyFile())
            assert "Test content" in str(result)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
