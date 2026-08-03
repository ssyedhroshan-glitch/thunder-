import os
import sys
import uuid
import tempfile
import sqlite3
import io
import re
import json
import base64
import contextlib
import requests

# ==========================================
# GRADIO SCHEMA BUG PATCH
# ==========================================
import gradio_client.utils as _gc_utils

_original_get_type = _gc_utils.get_type
def _patched_get_type(schema):
    if isinstance(schema, bool): return "Any"
    return _original_get_type(schema)
_gc_utils.get_type = _patched_get_type

_original_json_schema_to_python_type = _gc_utils._json_schema_to_python_type
def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool): return "Any"
    return _original_json_schema_to_python_type(schema, defs)
_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type

import gradio as gr
from huggingface_hub import InferenceClient

# ==========================================
# ENVIRONMENT & CLIENT CONFIGURATION
# ==========================================
hf_token = os.environ.get("HF_TOKEN")
openai_key = os.environ.get("OPENAI_API_KEY")
gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
ollama_key = os.environ.get("OLLAMA_API_KEY")
ollama_host = os.environ.get("OLLAMA_HOST", "https://ollama.com")

DB_PATH = "thunder_memory.db"

# Fast-loading Clients
qwen_client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=hf_token, timeout=15) if hf_token else InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", timeout=15)
whisper_client = InferenceClient(model="openai/whisper-large-v3", token=hf_token, timeout=15) if hf_token else InferenceClient(model="openai/whisper-large-v3", timeout=15)

# ==========================================
# LEVEL 2: FAST AUTONOMOUS TOOLS
# ==========================================
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform quick web searches to look up up-to-date facts.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Search query."}},
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code in an isolated environment.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Python snippet."}},
                "required": ["code"]
            }
        }
    }
]

def tool_web_search(query: str):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))
        if not results: return "No web results found."
        return "\n---\n".join([f"Title: {r.get('title')}\nURL: {r.get('href')}\nSnippet: {r.get('body')}" for r in results])
    except Exception as e:
        return f"Web Search Error: {e}"

def tool_execute_python(code: str):
    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            exec_globals = {"__builtins__": __builtins__}
            exec(code, exec_globals)
        res = output_buffer.getvalue()
        return res if res.strip() else "[Code executed successfully]"
    except Exception as e:
        return f"Python Execution Error: {e}"

def execute_tool_call(tool_name, arguments):
    if tool_name == "web_search": return tool_web_search(arguments.get("query", ""))
    elif tool_name == "execute_python": return tool_execute_python(arguments.get("code", ""))
    return f"Unknown tool: {tool_name}"

# ==========================================
# DATABASE & STORAGE MANAGEMENT
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, name TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
    conn.execute("CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, role TEXT, content TEXT)")
    cursor = conn.cursor()
    if cursor.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0:
        cursor.execute("INSERT INTO sessions (id, name) VALUES (?, ?)", (str(uuid.uuid4()), "Default Workspace"))
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
    conn.execute("INSERT INTO history (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, str(content)))
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

# ==========================================
# FAST MEDIA ENGINES (BASE64 & WHISPER)
# ==========================================
def generate_image_flux_base64(prompt: str):
    """Generates FLUX images directly into base64 data URIs so Gradio/HTML renders them reliably."""
    try:
        flux_client = InferenceClient(model="black-forest-labs/FLUX.1-schnell", token=hf_token, timeout=30)
        image_obj = flux_client.text_to_image(prompt)
        buffered = io.BytesIO()
        image_obj.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}", None
    except Exception as e:
        return None, str(e)

def transcribe_audio(audio_path):
    if not audio_path: return ""
    try:
        res = whisper_client.automatic_speech_recognition(audio_path)
        return res.get("text", "") if isinstance(res, dict) else str(res)
    except Exception as e:
        return f"[Audio Error: {e}]"

def read_file(file_obj):
    if file_obj is None: return ""
    try:
        file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n".join((page.extract_text() or "") for i, page in enumerate(reader.pages) if i < 10)[:6000]
        else:
            with open(file_path, "r", errors="ignore", encoding="utf-8") as f:
                return f.read(8000)[:6000]
    except Exception as e:
        return f"[File Read Error: {e}]"

# ==========================================
# LEVEL 3 ARTIFACT EXTRACTOR
# ==========================================
def extract_artifacts(text):
    python_matches = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    html_matches = re.findall(r"```html\s*(.*?)\s*```", text, re.DOTALL)
    
    python_code = python_matches[-1].strip() if python_matches else None
    html_code = html_matches[-1].strip() if html_matches else None
    
    return python_code, html_code

# ==========================================
# STREAMING & MODEL ROUTER ENGINE
# ==========================================
def stream_model_response(model_choice, messages, temp, tokens):
    if "Gemini" in model_choice:
        current_gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not current_gemini_key:
            yield "⚠️ **Gemini Error:** `GEMINI_API_KEY` missing."
            return
        try:
            from google import genai
            g_client = genai.Client(api_key=current_gemini_key)
            formatted = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
            res = g_client.models.generate_content(model="gemini-1.5-flash", contents=formatted)
            yield res.text or "[Done]"
            return
        except Exception as e:
            yield f"⚠️ **Gemini API Error:** {e}"
            return

    elif "Ollama" in model_choice:
        if not ollama_key:
            yield "⚠️ **Ollama Error:** `OLLAMA_API_KEY` missing."
            return
        try:
            prompt_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
            resp = requests.post(
                f"{ollama_host}/api/generate",
                headers={"Authorization": f"Bearer {ollama_key}"},
                json={"model": "llama3.2", "prompt": prompt_text, "stream": False},
                timeout=15
            )
            yield resp.json().get("response", "[No response]")
            return
        except Exception as e:
            yield f"⚠️ **Ollama Error:** {e}"
            return

    else:
        # Fast Streaming Qwen Engine
        try:
            full_res = ""
            for token in qwen_client.chat_completion(messages, max_tokens=int(tokens), temperature=float(temp), stream=True):
                chunk = token.choices[0].delta.content
                if chunk:
                    full_res += chunk
                    yield full_res
            return
        except Exception as e:
            yield f"⚠️ **Model Error:** {e}"

# ==========================================
# OPTIMIZED GRADIO UI
# ==========================================
DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder AI. Generate direct, accurate responses and code snippets. "
    "When asked to write HTML or Web UI, format it inside ```html ... ``` blocks."
)

custom_css = """
footer {visibility: hidden;}
body, .gradio-container {background-color: #0b0f19 !important;}
.panel-card {
    background-color: #121826 !important;
    border: 1px solid #1f293d !important;
    border-radius: 12px !important;
    padding: 12px !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"), css=custom_css) as demo:
    session_id = gr.State(None)
    chat_state = gr.State([])
    file_context_state = gr.State("")

    gr.Markdown("<center><h2 style='color:#22d3ee;'>⚡ THUNDER AI — OPTIMIZED LEVEL 1, 2 & 3</h2></center>")

    with gr.Row():
        # LEFT PANEL: Controls & Input
        with gr.Column(scale=3, elem_classes=["panel-card"]):
            session_selector = gr.Dropdown(choices=[], label="Workspace Session")
            new_session_btn = gr.Button("➕ New Workspace", size="sm")
            model_choice = gr.Dropdown(
                choices=[
                    "Qwen 2.5 7B (Fast Stream)",
                    "Gemini 1.5 Flash",
                    "Ollama API",
                    "DeepSeek R1 (Reasoning)"
                ],
                value="Qwen 2.5 7B (Fast Stream)",
                label="AI Core Model"
            )
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Directives", lines=2)
            temperature = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
            max_tokens = gr.Slider(128, 2048, value=1024, label="Max Tokens")
            
            file_input = gr.File(label="Attach File (PDF/TXT)", file_count="single")
            # Fixed Mic configuration for mobile and web browsers
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎙️ Voice Input (Microphone)")
            clear_btn = gr.Button("Reset Chat", variant="stop")

        # CENTER PANEL: Chatbot
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(height=520, type="messages")
            with gr.Row():
                msg = gr.Textbox(placeholder="Ask anything, 'generate an image of...', or 'build an HTML app'...", show_label=False, scale=9)
                send_btn = gr.Button("⚡ Run", variant="primary", scale=1)

        # RIGHT PANEL: Level 3 Interactive Workspace
        with gr.Column(scale=5, elem_classes=["panel-card"]):
            gr.Markdown("### 🎨 Level 3 Interactive Artifact Workspace")
            
            with gr.Tabs():
                with gr.TabItem("🌐 Live Web Preview"):
                    html_preview = gr.HTML(value="<div style='color:#94a3b8; text-align:center; padding:20px;'>HTML preview renders here instantly.</div>")
                
                with gr.TabItem("🐍 Python Sandbox"):
                    py_code_box = gr.Code(label="Extracted Python Code", language="python", lines=10)
                    exec_py_btn = gr.Button("▶️ Run Code", variant="primary", size="sm")
                    py_out_box = gr.Textbox(label="Output", lines=4, interactive=False)

    # Session Database Handlers
    def start_session():
        init_db()
        sessions = get_all_sessions()
        active_id = sessions[0][0] if sessions else create_new_session()
        hist = load_history(active_id)
        return active_id, hist, hist, gr.update(choices=[(n, s) for s, n in sessions], value=active_id)

    demo.load(start_session, None, [session_id, chatbot, chat_state, session_selector])

    def switch_session(target_id):
        if not target_id: return gr.update(), [], []
        hist = load_history(target_id)
        return target_id, hist, hist

    session_selector.change(switch_session, inputs=[session_selector], outputs=[session_id, chatbot, chat_state])

    def handle_new_session():
        new_id = create_new_session("New Workspace")
        sessions = get_all_sessions()
        return new_id, [], [], gr.update(choices=[(n, s) for s, n in sessions], value=new_id)

    new_session_btn.click(handle_new_session, None, [session_id, chatbot, chat_state, session_selector])

    audio_input.change(lambda path: transcribe_audio(path) if path else "", inputs=[audio_input], outputs=[msg])
    file_input.change(lambda file_obj: read_file(file_obj), inputs=[file_input], outputs=[file_context_state])

    def user_send(message, history, sid):
        if not message.strip(): return "", history, history
        new_hist = history + [{"role": "user", "content": message}]
        save_message(sid, "user", message)
        return "", new_hist, new_hist

    def bot_reply(history, sys_prompt, model_sel, temp, tokens, f_context, sid, current_py, current_html):
        if not history: yield history, history, current_py, current_html; return

        last_user_msg = history[-1]["content"].strip()

        # Image Generation Intent Detection
        image_patterns = [
            r"generate\s+(?:an?\s+)?image\s+(?:of\s+)?(.*)",
            r"create\s+(?:an?\s+)?image\s+(?:of\s+)?(.*)",
            r"draw\s+(.*)"
        ]
        
        image_prompt = None
        for pattern in image_patterns:
            match = re.search(pattern, last_user_msg, re.IGNORECASE)
            if match:
                image_prompt = match.group(1).strip()
                break

        if image_prompt:
            history = history + [{"role": "assistant", "content": f"🎨 **Generating image for:** *\"{image_prompt}\"*..."}]
            yield history, history, current_py, current_html
            
            base64_img, err = generate_image_flux_base64(image_prompt)
            if err:
                history[-1]["content"] = f"⚠️ **Image Generation Error:** {err}"
            else:
                history[-1]["content"] = f"Here is your generated image:\n\n<img src='{base64_img}' style='max-width:100%; border-radius:8px;' />"
                
            save_message(sid, "assistant", history[-1]["content"])
            yield history, history, current_py, current_html
            return

        # Prepare System Prompt & Document Context
        messages = [{"role": "system", "content": sys_prompt}]
        if f_context: 
            messages.append({"role": "system", "content": f"Document Context:\n{f_context}"})

        for msg_item in history:
            messages.append({"role": msg_item["role"], "content": msg_item["content"]})

        history = history + [{"role": "assistant", "content": ""}]

        full_text = ""
        for chunk in stream_model_response(model_sel, messages, temp, tokens):
            full_text = chunk
            history[-1]["content"] = full_text
            
            py_code, html_code = extract_artifacts(full_text)
            updated_py = py_code if py_code else current_py
            updated_html = html_code if html_code else current_html
            
            yield history, history, updated_py, updated_html

        save_message(sid, "assistant", full_text)

    msg.submit(user_send, [msg, chat_state, session_id], [msg, chatbot, chat_state]).then(
        bot_reply, 
        [chat_state, system_prompt, model_choice, temperature, max_tokens, file_context_state, session_id, py_code_box, html_preview], 
        [chatbot, chat_state, py_code_box, html_preview]
    )

    send_btn.click(user_send, [msg, chat_state, session_id], [msg, chatbot, chat_state]).then(
        bot_reply, 
        [chat_state, system_prompt, model_choice, temperature, max_tokens, file_context_state, session_id, py_code_box, html_preview], 
        [chatbot, chat_state, py_code_box, html_preview]
    )

    exec_py_btn.click(tool_execute_python, inputs=[py_code_box], outputs=[py_out_box])
    clear_btn.click(lambda sid: (clear_session_history(sid), [], [])[1:], inputs=[session_id], outputs=[chatbot, chat_state])

port_number = int(os.environ.get("PORT", 10000))
demo.queue(default_concurrency_limit=8).launch(server_name="0.0.0.0", server_port=port_number)
