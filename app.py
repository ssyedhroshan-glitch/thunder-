import os
import sys
import uuid
import tempfile
import sqlite3
import io
import re
import json
import contextlib
import requests

# --- GRADIO SCHEMA BUG PATCH ---
import gradio_client.utils as _gc_utils

_original_get_type = _gc_utils.get_type
def _patched_get_type(schema):
    if isinstance(schema, bool):
        return "Any"
    return _original_get_type(schema)
_gc_utils.get_type = _patched_get_type

_original_json_schema_to_python_type = _gc_utils._json_schema_to_python_type
def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any"
    return _original_json_schema_to_python_type(schema, defs)
_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
# --- END PATCH ---

import gradio as gr
from huggingface_hub import InferenceClient

hf_token = os.environ.get("HF_TOKEN")

# Core Clients
qwen_client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=hf_token)
whisper_client = InferenceClient(model="openai/whisper-large-v3", token=hf_token)

DB_PATH = "thunder_memory.db"

# ==========================================
# 1. AUTONOMOUS TOOL DEFINITIONS & EXECUTION
# ==========================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform live web searches to look up up-to-date facts, news, documentation, or real-time information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Run Python code in an isolated environment to perform math, data calculations, logic, or algorithmic problem-solving.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Valid Python code snippet to execute."}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_url",
            "description": "Scrape and read raw text content from a given web URL for in-depth analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The target website URL starting with http:// or https://"}
                },
                "required": ["url"]
            }
        }
    }
]

def tool_web_search(query: str):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=4))
        if not results:
            return "No relevant search results found."
        formatted = []
        for r in results:
            formatted.append(f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}\n")
        return "\n---\n".join(formatted)
    except Exception as e:
        return f"Search execution error: {e}"

def tool_execute_python(code: str):
    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            exec_globals = {"__builtins__": __builtins__}
            exec(code, exec_globals)
        res = output_buffer.getvalue()
        return res if res.strip() else "[Code Executed Successfully with no stdout output]"
    except Exception as e:
        return f"Python Execution Error: {e}"

def tool_scrape_url(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        text = re.sub(r'<[^>]+>', ' ', resp.text)
        text = ' '.join(text.split())
        return text[:4000] if text else "Page loaded but no text content found."
    except Exception as e:
        return f"URL Scraping Error: {e}"

def execute_tool_call(tool_name, arguments):
    """Router for function calls."""
    if tool_name == "web_search":
        return tool_web_search(arguments.get("query", ""))
    elif tool_name == "execute_python":
        return tool_execute_python(arguments.get("code", ""))
    elif tool_name == "scrape_url":
        return tool_scrape_url(arguments.get("url", ""))
    else:
        return f"Unknown tool: {tool_name}"


# ==========================================
# 2. PERSISTENT SQL DATABASE ENGINE
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON history(session_id)")

    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sessions")
    if cursor.fetchone()[0] == 0:
        default_id = str(uuid.uuid4())
        cursor.execute("INSERT INTO sessions (id, name) VALUES (?, ?)", (default_id, "Default Workspace"))
    conn.commit()
    conn.close()

def get_all_sessions():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id, name FROM sessions ORDER BY created_at DESC").fetchall()
    conn.close()
    return rows

def create_new_session(name="New Workspace"):
    conn = sqlite3.connect(DB_PATH)
    new_id = str(uuid.uuid4())
    conn.execute("INSERT INTO sessions (id, name) VALUES (?, ?)", (new_id, name))
    conn.commit()
    conn.close()
    return new_id

def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO history (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

def clear_session_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def load_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT role, content FROM history WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

init_db()

DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder AI, an autonomous multi-modal co-founder and AI platform. "
    "You have access to tools: web_search, execute_python, and scrape_url. "
    "Use tools autonomously when you need factual information, exact code/math validation, or web scraping."
)

# ==========================================
# 3. AUTONOMOUS AGENT REASONING LOOP
# ==========================================
def run_autonomous_agent(model_choice, messages, temp, max_tokens):
    """
    Executes an autonomous agent loop: 
    Model -> Tool Call Request -> Execution -> Feedback Observation -> Final LLM Response.
    """
    max_loops = 3
    current_loop = 0

    while current_loop < max_loops:
        current_loop += 1
        
        # Call model with available tool definitions
        try:
            response = qwen_client.chat_completion(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=int(max_tokens),
                temperature=float(temp)
            )
            
            message_obj = response.choices[0].message
            
            # Check if model requested a tool call
            if hasattr(message_obj, "tool_calls") and message_obj.tool_calls:
                for tool_call in message_obj.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except:
                        args = {}
                        
                    yield f"🛠️ **Agent Action:** Calling tool `{fn_name}` with args `{args}`...\n\n"
                    
                    # Execute tool
                    tool_result = execute_tool_call(fn_name, args)
                    
                    # Append tool response back to message history
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": json.dumps(args)}
                        }]
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(tool_result)
                    })
            else:
                # No tool needed; yield final text
                yield message_obj.content or "[Completed]"
                return

        except Exception as e:
            # Fallback for models or API errors without strict tool compatibility
            yield f"⚠️ Agent Loop Notice: Switching to direct inference mode ({e})...\n\n"
            full_res = ""
            for token in qwen_client.chat_completion(messages, max_tokens=int(max_tokens), temperature=float(temp), stream=True):
                chunk = token.choices[0].delta.content
                if chunk:
                    full_res += chunk
                    yield full_res
            return


# ==========================================
# 4. GRADIO UI INTERFACE
# ==========================================
custom_css = """
footer {visibility: hidden;}
body, .gradio-container {background-color: #0b0f19 !important;}
.panel-card {
    background-color: #121826 !important;
    border: 1px solid #1f293d !important;
    border-radius: 12px !important;
    padding: 14px !important;
}
.chatbot-container {
    border: 1px solid #1f293d !important;
    border-radius: 12px !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"), css=custom_css) as demo:

    session_id = gr.State(None)
    chat_state = gr.State([])

    gr.Markdown("<center><h2 style='color:#22d3ee; margin-bottom:5px;'>⚡ THUNDER AI — LEVEL 2 AUTONOMOUS AGENT</h2></center>")

    with gr.Row():
        with gr.Column(scale=4, elem_classes=["panel-card"]):
            session_selector = gr.Dropdown(choices=[], label="Workspace Session", interactive=True)
            new_session_btn = gr.Button("➕ New Workspace", size="sm")
            
            model_choice = gr.Dropdown(
                choices=["Qwen 2.5 7B Agent", "Gemini 1.5 Flash"],
                value="Qwen 2.5 7B Agent",
                label="Select AI Model Engine"
            )
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Instructions", lines=3)
            
            with gr.Row():
                temperature = gr.Slider(0.1, 1.5, value=0.70, step=0.05, label="Temperature")
                max_tokens = gr.Slider(256, 4096, value=1536, step=128, label="Max Tokens")

            clear_btn = gr.Button("Reset Chat", variant="stop")

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(height=560, elem_classes=["chatbot-container"], type="messages")
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask Thunder AI anything or give it a complex multi-step task...", show_label=False, container=False, scale=9)
                send_btn = gr.Button("⚡ Run Agent", variant="primary", scale=1)

    def start_session():
        init_db()
        sessions = get_all_sessions()
        active_id = sessions[0][0] if sessions else create_new_session()
        hist = load_history(active_id)
        choices = [(name, sid) for sid, name in sessions]
        return active_id, hist, hist, gr.update(choices=choices, value=active_id)

    demo.load(start_session, None, [session_id, chatbot, chat_state, session_selector])

    def switch_session(target_id):
        if not target_id: return gr.update(), [], []
        hist = load_history(target_id)
        return target_id, hist, hist

    session_selector.change(switch_session, inputs=[session_selector], outputs=[session_id, chatbot, chat_state])

    def handle_new_session():
        new_id = create_new_session("New Workspace")
        sessions = get_all_sessions()
        choices = [(name, sid) for sid, name in sessions]
        return new_id, [], [], gr.update(choices=choices, value=new_id)

    new_session_btn.click(handle_new_session, None, [session_id, chatbot, chat_state, session_selector])

    def user_send(message, history, sid):
        if not message.strip(): return "", history, history
        new_hist = history + [{"role": "user", "content": message}]
        save_message(sid, "user", message)
        return "", new_hist, new_hist

    def bot_agent_reply(history, sys_prompt, model_sel, temp, tokens, sid):
        if not history: yield history, history; return

        messages = [{"role": "system", "content": sys_prompt}]
        for msg_item in history:
            messages.append({"role": msg_item["role"], "content": msg_item["content"]})

        history = history + [{"role": "assistant", "content": ""}]

        accumulated_text = ""
        for agent_step in run_autonomous_agent(model_sel, messages, temp, tokens):
            accumulated_text += agent_step
            history[-1]["content"] = accumulated_text
            yield history, history

        save_message(sid, "assistant", accumulated_text)

    msg.submit(user_send, [msg, chat_state, session_id], [msg, chatbot, chat_state]).then(
        bot_agent_reply, 
        [chat_state, system_prompt, model_choice, temperature, max_tokens, session_id], 
        [chatbot, chat_state]
    )

    send_btn.click(user_send, [msg, chat_state, session_id], [msg, chatbot, chat_state]).then(
        bot_agent_reply, 
        [chat_state, system_prompt, model_choice, temperature, max_tokens, session_id], 
        [chatbot, chat_state]
    )

    clear_btn.click(lambda sid: (clear_session_history(sid), [], [])[1:], inputs=[session_id], outputs=[chatbot, chat_state])

port_number = int(os.environ.get("PORT", 10000))
demo.queue(default_concurrency_limit=4).launch(server_name="0.0.0.0", server_port=port_number)
                       
