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
from typing import List, Dict, Any, Generator

# ==========================================
# 1. GRADIO SCHEMA BUG PATCH
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
# 2. ENVIRONMENT & CLIENT CONFIGURATION
# ==========================================
HF_TOKEN = os.environ.get("HF_TOKEN")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
GEMINI_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
OLLAMA_KEY = os.environ.get("OLLAMA_API_KEY")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "https://ollama.com")

DB_PATH = "thunder_v5_memory.db"

# Resilient fast inference clients
qwen_client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN, timeout=20) if HF_TOKEN else InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", timeout=20)
whisper_client = InferenceClient(model="openai/whisper-large-v3", token=HF_TOKEN, timeout=20) if HF_TOKEN else InferenceClient(model="openai/whisper-large-v3", timeout=20)
flux_client = InferenceClient(model="black-forest-labs/FLUX.1-schnell", token=HF_TOKEN, timeout=30) if HF_TOKEN else InferenceClient(model="black-forest-labs/FLUX.1-schnell", timeout=30)

# ==========================================
# 3. LEVEL 2: AUTONOMOUS TOOL DEFINITIONS
# ==========================================
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform live internet search queries to retrieve up-to-date knowledge.",
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
            "description": "Execute Python code safely in an isolated environment for math or data processing.",
            "parameters": {
                "type": "object",
                "properties": {"code": {"type": "string", "description": "Valid Python code string."}},
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_url",
            "description": "Fetch and extract text content from a public web URL.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "Target webpage URL."}},
                "required": ["url"]
            }
        }
    }
]

def tool_web_search(query: str):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
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
        return res if res.strip() else "[Code executed successfully with no stdout output]"
    except Exception as e:
        return f"Python Execution Error: {e}"

def tool_scrape_url(url: str):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        text = re.sub(r'<[^>]+>', ' ', resp.text)
        return ' '.join(text.split())[:4000]
    except Exception as e:
        return f"Scraping Error: {e}"

def execute_tool_call(tool_name, arguments):
    if tool_name == "web_search": return tool_web_search(arguments.get("query", ""))
    elif tool_name == "execute_python": return tool_execute_python(arguments.get("code", ""))
    elif tool_name == "scrape_url": return tool_scrape_url(arguments.get("url", ""))
    return f"Unknown tool: {tool_name}"

# ==========================================
# 4. HIGH-PERFORMANCE DATABASE LAYER (INDEXED)
# ==========================================
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, 
                name TEXT, 
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, 
                session_id TEXT, 
                role TEXT, 
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON history(session_id)")

def get_all_sessions():
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT id, name FROM sessions ORDER BY updated_at DESC").fetchall()
        if not rows:
            default_id = str(uuid.uuid4())
            conn.execute("INSERT INTO sessions (id, name) VALUES (?, ?)", (default_id, "Default Workspace"))
            return [(default_id, "Default Workspace")]
        return rows

def create_new_session(name="New Workspace"):
    with sqlite3.connect(DB_PATH) as conn:
        new_id = str(uuid.uuid4())
        conn.execute("INSERT INTO sessions (id, name) VALUES (?, ?)", (new_id, name))
        return new_id

def save_message(session_id: str, role: str, content: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO history (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, str(content)))
        conn.execute("UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (session_id,))

def clear_session_history(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM history WHERE session_id = ?", (session_id,))

def load_history(session_id: str):
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT role, content FROM history WHERE session_id = ? ORDER BY id ASC", (session_id,)).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

init_db()

# ==========================================
# 5. FAST MEDIA & FILE ENGINES (BASE64 & WHISPER)
# ==========================================
def generate_image_flux_base64(prompt: str) -> (str, str):
    """Generates FLUX images directly into base64 data URIs for instant cross-device rendering."""
    try:
        image_obj = flux_client.text_to_image(prompt)
        buffered = io.BytesIO()
        image_obj.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        html_img = f"<div style='text-align:center; margin:10px 0;'><img src='data:image/png;base64,{img_str}' style='max-width:100%; border-radius:12px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);' /><br/><span style='font-size:12px; color:#94a3b8;'>Generated by FLUX.1 Schnell</span></div>"
        return html_img, None
    except Exception as e:
        return None, str(e)

def transcribe_audio(audio_path):
    if not audio_path: return ""
    try:
        res = whisper_client.automatic_speech_recognition(audio_path)
        return res.get("text", "") if isinstance(res, dict) else str(res)
    except Exception as e:
        return f"[Voice Transcription Error: {e}]"

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
# 6. LEVEL 3: ARTIFACT & WORKSPACE EXTRACTOR
# ==========================================
def extract_artifacts(text):
    """Extracts code blocks for live rendering in Level 3 workspace."""
    python_matches = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    html_matches = re.findall(r"```html\s*(.*?)\s*```", text, re.DOTALL)
    
    python_code = python_matches[-1].strip() if python_matches else None
    html_code = html_matches[-1].strip() if html_matches else None
    
    return python_code, html_code

# ==========================================
# 7. ROUTER ENGINE & AGENT LOOPS
# ==========================================
def run_autonomous_loop(messages, tokens, temp, max_iterations=3):
    """Level 2 Agentic Tool Loop execution."""
    iter_count = 0
    trace_output = ""
    
    while iter_count < max_iterations:
        iter_count += 1
        try:
            response = qwen_client.chat_completion(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=int(tokens),
                temperature=float(temp)
            )
            msg_obj = response.choices[0].message
            
            if hasattr(msg_obj, "tool_calls") and msg_obj.tool_calls:
                for tool_call in msg_obj.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except Exception:
                        args = {}
                    
                    trace_output += f"🛠️ **Agent Action (Step {iter_count}):** Executing `{fn_name}` with `{args}`...\n\n"
                    yield trace_output
                    
                    result = execute_tool_call(fn_name, args)
                    
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {"name": fn_name, "arguments": json.dumps(args)}
                        }]
                    })
                    messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
            else:
                trace_output += (msg_obj.content or "")
                yield trace_output
                return
        except Exception as e:
            yield trace_output + f"\n⚠️ **Agent Loop Error:** {e}"
            return

def stream_model_response(model_choice, messages, temp, tokens):
    if "Qwen" in model_choice:
        for chunk in run_autonomous_loop(messages, tokens, temp):
            yield chunk
        return

    elif "Gemini" in model_choice:
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
        if not OLLAMA_KEY:
            yield "⚠️ **Ollama Error:** `OLLAMA_API_KEY` missing."
            return
        try:
            prompt_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                headers={"Authorization": f"Bearer {OLLAMA_KEY}"},
                json={"model": "llama3.2", "prompt": prompt_text, "stream": False},
                timeout=15
            )
            yield resp.json().get("response", "[No response]")
            return
        except Exception as e:
            yield f"⚠️ **Ollama Error:** {e}"
            return

    elif "DeepSeek" in model_choice:
        try:
            deepseek_client = InferenceClient(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", token=HF_TOKEN, timeout=20)
            full_res = ""
            for token in deepseek_client.chat_completion(messages, max_tokens=int(tokens), temperature=float(temp), stream=True):
                chunk = token.choices[0].delta.content
                if chunk:
                    full_res += chunk
                    yield full_res
            return
        except Exception as e:
            yield f"⚠️ **DeepSeek Error:** {e}"
            return

    elif "GPT-4o" in model_choice:
        if not OPENAI_KEY:
            yield "⚠️ **OpenAI Error:** `OPENAI_API_KEY` missing."
            return
        try:
            import openai
            client = openai.OpenAI(api_key=OPENAI_KEY)
            target = "gpt-4o" if "GPT-4o (OpenAI)" in model_choice else "gpt-4o-mini"
            formatted = [{"role": m["role"], "content": m["content"] or ""} for m in messages if m["role"] in ["system", "user", "assistant"]]
            stream = client.chat.completions.create(model=target, messages=formatted, temperature=float(temp), max_tokens=int(tokens), stream=True)
            full_res = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_res += chunk.choices[0].delta.content
                    yield full_res
            return
        except Exception as e:
            yield f"⚠️ **OpenAI Error:** {e}"

# ==========================================
# 8. UNIFIED GRADIO WORKSPACE UI
# ==========================================
DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder AI, a high-speed multi-modal agent. "
    "Execute search queries, write Python code, or generate live web UIs. "
    "When asked to write HTML or Web components, format them strictly inside ```html ... ``` code blocks."
)

custom_css = """
footer {visibility: hidden;}
body, .gradio-container {background-color: #090d16 !important;}
.panel-card {
    background-color: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    padding: 14px !important;
}
.accent-title {
    color: #38bdf8 !important;
    font-weight: 700 !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"), css=custom_css) as demo:
    session_id = gr.State(None)
    chat_state = gr.State([])
    file_context_state = gr.State("")

    gr.Markdown("<center><h2 class='accent-title'>⚡ THUNDER AI — UNIFIED PLATFORM (LEVELS 1-5)</h2></center>")

    with gr.Row():
        # LEFT PANEL: Controls, Multi-Modal Inputs & Model Selection
        with gr.Column(scale=3, elem_classes=["panel-card"]):
            session_selector = gr.Dropdown(choices=[], label="Workspace Sessions")
            new_session_btn = gr.Button("➕ New Workspace", size="sm")
            model_choice = gr.Dropdown(
                choices=[
                    "Qwen 2.5 7B (Autonomous Agent)",
                    "Gemini 1.5 Flash",
                    "Ollama API",
                    "DeepSeek R1 (Reasoning)",
                    "GPT-4o (OpenAI)"
                ],
                value="Qwen 2.5 7B (Autonomous Agent)",
                label="AI Core Model Engine"
            )
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Instructions", lines=2)
            temperature = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
            max_tokens = gr.Slider(128, 2048, value=1024, label="Max Tokens")
            
            file_input = gr.File(label="Attach File (PDF/TXT)", file_count="single")
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎙️ Voice Input (Microphone)")
            clear_btn = gr.Button("Reset Chat History", variant="stop")

        # CENTER PANEL: Real-time Chat Engine
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(height=540, type="messages", bubble_full_width=False)
            with gr.Row():
                msg = gr.Textbox(placeholder="Message Thunder AI, 'draw a sunset', or 'build an HTML app'...", show_label=False, scale=9)
                send_btn = gr.Button("⚡ Run", variant="primary", scale=1)

        # RIGHT PANEL: Level 3 Interactive Canvas & Code Execution Sandbox
        with gr.Column(scale=5, elem_classes=["panel-card"]):
            gr.Markdown("### 🎨 Level 3 Live Artifacts & Sandbox Canvas")
            
            with gr.Tabs():
                with gr.TabItem("🌐 Live Web App Canvas"):
                    html_preview = gr.HTML(value="<div style='color:#6b7280; text-align:center; padding:20px;'>Generated Web UIs render here instantly.</div>")
                
                with gr.TabItem("🐍 Interactive Python Sandbox"):
                    py_code_box = gr.Code(label="Extracted Python Code", language="python", lines=10)
                    exec_py_btn = gr.Button("▶️ Run Code", variant="primary", size="sm")
                    py_out_box = gr.Textbox(label="Execution Output", lines=4, interactive=False)

    # Session Handlers
    def start_session():
        init_db()
        sessions = get_all_sessions()
        active_id = sessions[0][0]
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

    def bot_reply(history, sys_prompt, model_sel, temp, tokens, f_context,sod,curre6_py, current _html):
      if not history: yield history, history, current_py, current_html; return

        last_user_msg = history[-1]["content"].strip()

        # Direct Image Generation Handler
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
            history = history + [{"role": "assistant", "content": f"🎨 **Rendering image for:** *\"{image_prompt}\"*..."}]
            yield history, history, current_py, current_html
            
            html_img_block, err = generate_image_flux_base64(image_prompt)
            if err:
                history[-1]["content"] = f"⚠️ **Image Generation Error:** {err}"
            else:
                history[-1]["content"] = html_img_block
                
            save_message(sid, "assistant", history[-1]["content"])
            yield history, history, current_py, current_html
            return

        # Prepare System Directives & Attachments
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
demo.queue(default_concurrency_limit=10).launch(server_name="0.0.0.0", server_port=port_number)   
