import os
import sqlite3
import pytest
import tempfile
from app import (
    init_db,
    create_new_session,
    get_all_sessions,
    rename_session_in_db,
    delete_session_from_db,
    save_message,
    load_history,
    execute_python_code,
    read_file,
    DB_PATH
)

@pytest.fixture(autouse=True)
def setup_and_teardown_db():
    """Ensure a clean database state for every test."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    init_db()
    yield
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def test_db_initialization():
    """Verify default session creation on clean database setup."""
    sessions = get_all_sessions()
    assert len(sessions) == 1
    assert sessions[0][1] == "Default Workspace"

def test_session_lifecycle():
    """Test creating, renaming, and deleting sessions."""
    # Create
    new_id = create_new_session("Dev Environment")
    sessions = get_all_sessions()
    assert len(sessions) == 2
    
    # Rename
    rename_session_in_db(new_id, "Staging Environment")
    updated_sessions = get_all_sessions()
    session_names = [s[1] for s in updated_sessions]
    assert "Staging Environment" in session_names

    # Delete
    delete_session_from_db(new_id)
    final_sessions = get_all_sessions()
    assert len(final_sessions) == 1

def test_message_persistence():
    """Verify saving and retrieving chat history for a session."""
    sessions = get_all_sessions()
    session_id = sessions[0][0]

    save_message(session_id, "user", "Hello Thunder")
    save_message(session_id, "assistant", "Systems online.")

    history = load_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello Thunder"
    assert history[1]["role"] == "assistant"

def test_code_sandbox_execution():
    """Test standard execution and error handling in the Python sandbox."""
    success_output = execute_python_code("print(2 + 2)")
    assert "4" in success_output

    error_output = execute_python_code("1 / 0")
    assert "ZeroDivisionError" in error_output

def test_file_reader_text_file():
    """Verify parsing plain text files."""
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False, encoding="utf-8") as tmp:
        tmp.write("Thunder Workspace context line.")
        tmp_path = tmp.name

    try:
        class DummyFileObj:
            name = tmp_path

        parsed_content = read_file(DummyFileObj())
        assert "Thunder Workspace context line." in parsed_content
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
          
