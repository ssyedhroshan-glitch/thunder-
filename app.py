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

# Environment Variables
hf_token = os.environ.get("HF_TOKEN")
openai_key = os.environ.get("OPENAI_API_KEY")
gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
ollama_key = os.environ.get("OLLAMA_API_KEY")
ollama_host = os.environ.get("OLLAMA_HOST", "https://ollama.com")

# Initialize Hugging Face Clients
qwen_client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=hf_token) if hf_token else InferenceClient(model="Qwen/Qwen2.5-7B-Instruct")
whisper_client = InferenceClient(model="openai/whisper-large-v3", token=hf_token) if hf_token else InferenceClient(model="openai/whisper-large-v3")

DB_PATH = "thunder_memory.db"

# ==========================================
# 1. AUTONOMOUS TOOL DEFINITIONS & EXECUTION
# ==========================================

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Perform live web searches to look up real-time information, news, or documentation.",
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
            "description": "Execute Python code in an isolated environment to handle math, logic, or calculations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Valid Python code snippet."}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scrape_url",
            "description": "Fetch text content from a web URL for analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Target web URL."}
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
        headers = {"User-Agent": "Mozilla/5.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        text = re.sub(r'<[^>]+>', ' ', resp.text)
        text = ' '.join(text.split())
        return text[:4000] if text else "No printable text content found."
    except Exception as e:
        return f"Scraping Error: {e}"

def execute_tool_call(tool_name, arguments):
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

DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder AI, an elite multi-modal assistant and autonomous agent platform. "
    "You can answer questions, reason through code, execute Python, and generate detailed visual imagery."
)

# ==========================================
# 3. HELPER FUNCTIONS (WHISPER, PDF, IMAGE)
# ==========================================
def extract_python_code(text):
    matches = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

def generate_image_flux(prompt: str):
    """Generates images using FLUX.1-schnell via Hugging Face Client."""
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
            text = "\n".join((page.extract_text() or "") for i, page in enumerate(reader.pages) if i < 15)
        else:
            with open(file_path, "r", errors="ignore", encoding="utf-8") as f:
                text = f.read(15000)
        return text[:12000]
    except Exception as e:
        return f"[File Parsing Error: {e}]"


# ==========================================
# 4. MULTI-MODEL ROUTER
# ==========================================
def stream_model_response(model_choice, messages, temp, tokens):
    # --- OPENAI HANDLER ---
    if "GPT-4o" in model_choice:
        if not os.environ.get("OPENAI_API_KEY"):
            yield "⚠️ **OpenAI Error:** `OPENAI_API_KEY` environment variable is not configured on Render."
            return
        try:
            import openai
            client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            target_model = "gpt-4o" if "GPT-4o (OpenAI)" in model_choice else "gpt-4o-mini"
            
            formatted_messages = []
            for m in messages:
                if m["role"] in ["system", "user", "assistant"]:
                    formatted_messages.append({"role": m["role"], "content": m["content"] or ""})

            response = client.chat.completions.create(
                model=target_model,
                messages=formatted_messages,
                temperature=float(temp),
                max_tokens=int(tokens),
                stream=True
            )
            full_res = ""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    full_res += chunk.choices[0].delta.content
                    yield full_res
            return
        except Exception as e:
            yield f"⚠️ **OpenAI API Error:** {e}"
            return

    # --- GEMINI HANDLER ---
    elif model_choice == "Gemini 1.5 Flash":
        current_gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not current_gemini_key:
            yield "⚠️ **Gemini Error:** `GEMINI_API_KEY` is not set in Render environment variables."
            return
        
        # Method 1: New google-genai SDK
        try:
            from google import genai
            g_client = genai.Client(api_key=current_gemini_key)
            
            system_instr = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
            user_contents = []
            for m in messages:
                if m["role"] != "system":
                    user_contents.append({"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]})

            resp = g_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=user_contents,
                config={"system_instruction": system_instr, "temperature": float(temp), "max_output_tokens": int(tokens)}
            )
            yield resp.text or "[Response completed]"
            return
        except Exception as err1:
            # Method 2: Legacy google.generativeai SDK fallback
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=current_gemini_key)
                gen_model = legacy_genai.GenerativeModel("gemini-1.5-flash")
                prompt_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
                res = gen_model.generate_content(prompt_text)
                yield res.text
                return
            except Exception as err2:
                yield f"⚠️ **Gemini API Error:** {err1} | {err2}"
                return

    # --- OLLAMA API HANDLER ---
    elif model_choice == "Ollama API":
        if not ollama_key:
            yield "⚠️ **Ollama Error:** `OLLAMA_API_KEY` is not set in Render environment variables."
            return
        try:
            prompt_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in messages])
            resp = requests.post(
                f"{ollama_host}/api/generate",
                headers={"Authorization": f"Bearer {ollama_key}", "Content-Type": "application/json"},
                json={"model": "llama3.2", "prompt": prompt_text, "stream": False},
                timeout=30
            )
            if resp.status_code == 200:
                yield resp.json().get("response", "[No output from Ollama]")
            else:
                yield f"⚠️ **Ollama HTTP Error {resp.status_code}:** {resp.text}"
            return
        except Exception as e:
            yield f"⚠️ **Ollama Connection Error:** {e}"
            return

    # --- DEEPSEEK R1 REASONING HANDLER ---
    elif model_choice == "DeepSeek R1 (Reasoning)":
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
            yield f"⚠️ **DeepSeek R1 Error:** {e}"
            return

    # --- QWEN AUTONOMOUS AGENT HANDLER ---
    else:
        try:
            response = qwen_client.chat_completion(
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                max_tokens=int(tokens),
                temperature=float(temp)
            )
            message_obj = response.choices[0].message
            
            if hasattr(message_obj, "tool_calls") and message_obj.tool_calls:
                for tool_call in message_obj.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments)
                    except Exception:
                        args = {}
                    yield f"🛠️ **Agent Action:** Running `{fn_name}` with arguments `{args}`...\n\n"
                    tool_result = execute_tool_call(fn_name, args)
                    
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
                
                full_res = "### 📊 Tool Results & Output:\n"
                for token in qwen_client.chat_completion(messages, max_tokens=int(tokens), temperature=float(temp), stream=True):
                    chunk = token.choices[0].delta.content
                    if chunk:
                        full_res += chunk
                        yield full_res
                return
            else:
                yield message_obj.content or "[Completed]"
                return
        except Exception as e:
            full_res = ""
            for token in qwen_client.chat_completion(messages, max_tokens=int(tokens), temperature=float(temp), stream=True):
                chunk = token.choices[0].delta.content
                if chunk:
                    full_res += chunk
                    yield full_res


# ==========================================
# 5. GRADIO UI INTERFACE
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
    file_context_state = gr.State("")

    gr.Markdown("<center><h2 style='color:#22d3ee; margin-bottom:5px;'>⚡ THUNDER ADVANCED AI PLATFORM</h2></center>")

    with gr.Row():
        # LEFT SIDEBAR: Workspace, Models, Files, Voice
        with gr.Column(scale=4, elem_classes=["panel-card"]):
            session_selector = gr.Dropdown(choices=[], label="Workspace Session", interactive=True)
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
                label="Select AI Model Engine"
            )
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Instructions", lines=2)
            
            with gr.Row():
                temperature = gr.Slider(0.1, 1.5, value=0.70, step=0.05, label="Temperature")
                max_tokens = gr.Slider(256, 4096, value=1536, step=128, label="Max Tokens")

            file_input = gr.File(label="Attach Document (PDF/TXT)", file_count="single")
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input (Whisper)")
            clear_btn = gr.Button("Reset Chat", variant="stop")

        # CENTER: Main Chat Window
        with gr.Column(scale=8):
            chatbot = gr.Chatbot(height=520, elem_classes=["chatbot-container"], type="messages")
            with gr.Row():
                msg = gr.Textbox(placeholder="Message Thunder AI or ask 'generate an image of ...'", show_label=False, container=False, scale=9)
                send_btn = gr.Button("⚡ Run", variant="primary", scale=1)

        # RIGHT PANEL: Interactive Code Sandbox
        with gr.Column(scale=5, elem_classes=["panel-card"]):
            gr.Markdown("### 🛠️ Interactive Code Sandbox")
            artifact_code = gr.Code(label="Python Code Sandbox", language="python", lines=12)
            exec_btn = gr.Button("▶️ Execute Code", variant="primary", size="sm")
            exec_output = gr.Textbox(label="Execution Output", lines=5, interactive=False)

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

    audio_input.change(lambda path: transcribe(path) if path else "", inputs=[audio_input], outputs=[msg])
    file_input.change(lambda file_obj: read_file(file_obj), inputs=[file_input], outputs=[file_context_state])

    def user_send(message, history, sid):
        if not message.strip(): return "", history, history
        new_hist = history + [{"role": "user", "content": message}]
        save_message(sid, "user", message)
        return "", new_hist, new_hist

    def bot_reply(history, sys_prompt, model_sel, temp, tokens, f_context, sid, current_code):
        if not history: yield history, history, current_code; return

        last_user_msg = history[-1]["content"].strip()

        # Image Generation Intent Detection
        image_patterns = [
            r"generate\s+(?:an?\s+)?image\s+(?:of\s+)?(.*)",
            r"generator\s+(?:a|an)\s+image\s+(?:of\s+)?(.*)",
            r"create\s+(?:an?\s+)?image\s+(?:of\s+)?(.*)",
            r"draw\s+(.*)",
            r"make\s+(?:an?\s+)?image\s+(?:of\s+)?(.*)"
        ]
        
        image_prompt = None
        for pattern in image_patterns:
            match = re.search(pattern, last_user_msg, re.IGNORECASE)
            if match:
                image_prompt = match.group(1).strip()
                break

        if image_prompt:
            history = history + [{"role": "assistant", "content": f"🎨 **Generating image for:** *\"{image_prompt}\"*..."}]
            yield history, history, current_code
            
            img_path, err = generate_image_flux(image_prompt)
            if err:
                history[-1]["content"] = f"⚠️ **Image Generation Failed:** {err}"
            else:
                history[-1]["content"] = f"Here is your generated image:\n\n![{image_prompt}]({img_path})"
                
            save_message(sid, "assistant", history[-1]["content"])
            yield history, history, current_code
            return

        # Prepare System & Context Messages
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
            
            extracted_code = extract_python_code(full_text)
            code_out = extracted_code if extracted_code else current_code
            
            yield history, history, code_out

        save_message(sid, "assistant", full_text)

    msg.submit(user_send, [msg, chat_state, session_id], [msg, chatbot, chat_state]).then(
        bot_reply, 
        [chat_state, system_prompt, model_choice, temperature, max_tokens, file_context_state, session_id, artifact_code], 
        [chatbot, chat_state, artifact_code]
    )

    send_btn.click(user_send, [msg, chat_state, session_id], [msg, chatbot, chat_state]).then(
        bot_reply, 
        [chat_state, system_prompt, model_choice, temperature, max_tokens, file_context_state, session_id, artifact_code], 
        [chatbot, chat_state, artifact_code]
    )

    exec_btn.click(tool_execute_python, inputs=[artifact_code], outputs=[exec_output])
    clear_btn.click(lambda sid: (clear_session_history(sid), [], [])[1:], inputs=[session_id], outputs=[chatbot, chat_state])

port_number = int(os.environ.get("PORT", 10000))
demo.queue(default_concurrency_limit=4).launch(server_name="0.0.0.0", server_port=port_number)
