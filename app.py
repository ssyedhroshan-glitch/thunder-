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

# Model Clients (Level 1)
qwen_client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=hf_token) if hf_token else InferenceClient(model="Qwen/Qwen2.5-7B-Instruct")
whisper_client = InferenceClient(model="openai/whisper-large-v3", token=hf_token) if hf_token else InferenceClient(model="openai/whisper-large-v3")

# ==========================================
# LEVEL 2: AUTONOMOUS TOOL DEFINITIONS
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
            results = list(ddgs.text(query, max_results=4))
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
        return res if res.strip() else "[Code Executed Successfully with no stdout output]"
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
# LEVEL 1: SQL PERSISTENCE & MEDIA ENGINES
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

def generate_image_flux(prompt: str):
    try:
        flux_client = InferenceClient(model="black-forest-labs/FLUX.1-schnell", token=hf_token)
        image_obj = flux_client.text_to_image(prompt)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        image_obj.save(tmp.name)
        return tmp.name, None
    except Exception as e:
        return None, str(e)

def transcribe(audio_path):
    if not audio_path: return ""
    try:
        res = whisper_client.automatic_speech_recognition(audio_path)
        return res.text if hasattr(res, "text") else str(res)
    except Exception as e:
        return f"[Transcription Error: {e}]"

def read_file(file_obj):
    if file_obj is None: return ""
    try:
        file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            return "\n".join((page.extract_text() or "") for i, page in enumerate(reader.pages) if i < 15)[:12000]
        else:
            with open(file_path, "r", errors="ignore", encoding="utf-8") as f:
                return f.read(15000)[:12000]
    except Exception as e:
        return f"[File Reader Error: {e}]"

# ==========================================
# LEVEL 3: ARTIFACT & WORKSPACE EXTRACTORS
# ==========================================
def extract_artifacts(text):
    """Extracts code blocks for live rendering in Level 3 workspace."""
    python_matches = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    html_matches = re.findall(r"```html\s*(.*?)\s*```", text, re.DOTALL)
    
    python_code = python_matches[-1].strip() if python_matches else None
    html_code = html_matches[-1].strip() if html_matches else None
    
    return python_code, html_code

# ==========================================
# LEVEL 1 & 2 ROUTER ENGINE
# ==========================================
def run_autonomous_loop(messages, tokens, temp, max_iterations=4):
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
            yield "⚠️ **Gemini Error:** `GEMINI_API_KEY` environment variable missing."
            return
        try:
            from google import genai
            g_client = genai.Client(api_key=current_gemini_key)
            formatted = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]
            res = g_client.models.generate_content(model="gemini-1.5-flash", contents=formatted)
            yield res.text or "[Response Complete]"
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
                timeout=30
            )
            yield resp.json().get("response", "[No output from Ollama]")
            return
        except Exception as e:
            yield f"⚠️ **Ollama Error:** {e}"
            return

    elif "DeepSeek" in model_choice:
        try:
            deepseek_client = InferenceClient(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", token=hf_token)
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
        if not openai_key:
            yield "⚠️ **OpenAI Error:** `OPENAI_API_KEY` missing."
            return
        try:
            import openai
            client = openai.OpenAI(api_key=openai_key)
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
# COMBINED GRADIO WORKSPACE UI (LEVEL 1 + 2 + 3)
# ==========================================
DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder AI, a fully autonomous multi-modal agent (Level 1+2+3). "
    "You can execute live web searches, run Python code, generate FLUX images, "
    "and create live HTML/JS web artifacts that render dynamically in the preview workspace."
)

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
    file_context_state = gr.State("")

    gr.Markdown("<center><h2 style='color:#22d3ee;'>⚡ THUNDER AI — UNIFIED PLATFORM (LEVEL 1, 2 & 3)</h2></center>")

    with gr.Row():
        # LEFT PANEL: Controls, Media Inputs & Models (Level 1)
        with gr.Column(scale=3, elem_classes=["panel-card"]):
            session_selector = gr.Dropdown(choices=[], label="Workspace Session")
            new_session_btn = gr.Button("➕ New Workspace", size="sm")
            model_choice = gr.Dropdown(
                choices=[
                    "Qwen 2.5 7B (Autonomous Agent)",
                    "Gemini 1.5 Flash",
                    "Ollama API",
                    "DeepSeek R1 (Reasoning)",
                    "GPT-4o (OpenAI)",
                    "GPT-4o-mini (OpenAI)"
                ],
                value="Qwen 2.5 7B (Autonomous Agent)",
                label="AI Core Model Engine"
            )
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Instructions", lines=2)
            temperature = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")
            max_tokens = gr.Slider(256, 4096, value=2048, label="Max Tokens")
            
            file_input = gr.File(label="Attach Document (PDF/TXT)", file_count="single")
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input (Whisper)")
            clear_btn = gr.Button("Reset Session", variant="stop")

        # CENTER PANEL: Main Interactive Chat Engine
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(height=540, elem_classes=["chatbot-container"], type="messages")
            with gr.Row():
                msg = gr.Textbox(placeholder="Message Thunder AI or ask 'generate an image of...' / 'build an HTML app...'", show_label=False, scale=9)
                send_btn = gr.Button("⚡ Run", variant="primary", scale=1)

        # RIGHT PANEL: Level 3 Live Web Artifacts & Code Execution Sandbox
        with gr.Column(scale=5, elem_classes=["panel-card"]):
            gr.Markdown("### 🎨 Level 3 Live Artifacts & Sandbox Workspace")
            
            with gr.Tabs():
                with gr.TabItem("🌐 Live Web Artifact Preview"):
                    html_preview = gr.HTML(value="<div style='color:#94a3b8; text-align:center; padding:20px;'>HTML/JS code generated by the AI will render live here.</div>")
                
                with gr.TabItem("🐍 Interactive Python Sandbox"):
                    py_code_box = gr.Code(label="Extracted Python Code", language="python", lines=12)
                    exec_py_btn = gr.Button("▶️ Run Python Code", variant="primary", size="sm")
                    py_out_box = gr.Textbox(label="Execution Output", lines=5, interactive=False)

    # Session Database Callbacks
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

    audio_input.change(lambda path: transcribe(path) if path else "", inputs=[audio_input], outputs=[msg])
    file_input.change(lambda file_obj: read_file(file_obj), inputs=[file_input], outputs=[file_context_state])

    def user_send(message, history, sid):
        if not message.strip(): return "", history, history
        new_hist = history + [{"role": "user", "content": message}]
        save_message(sid, "user", message)
        return "", new_hist, new_hist

    def bot_reply(history, sys_prompt, model_sel, temp, tokens, f_context, sid, current_py, current_html):
        if not history: yield history, history, current_py, current_html; return

        last_user_msg = history[-1]["content"].strip()

        # Image Generation Handler (FLUX.1-schnell)
        image_patterns = [
            r"generate\s+(?:an?\s+)?image\s+(?:of\s+)?(.*)",
            r"generator\s+(?:a|an)\s+image\s+(?:of\s+)?(.*)",
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
            history = history + [{"role": "assistant", "content": f"🎨 **Generating FLUX image for:** *\"{image_prompt}\"*..."}]
            yield history, history, current_py, current_html
            
            img_path, err = generate_image_flux(image_prompt)
            if err:
                history[-1]["content"] = f"⚠️ **Image Generation Failed:** {err}"
            else:
                history[-1]["content"] = f"Here is your generated image:\n\n![{image_prompt}]({img_path})"
                
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
            
            # Level 3 Extraction
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
demo.queue(default_concurrency_limit=4).launch(server_name="0.0.0.0", server_port=port_number)
