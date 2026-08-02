import os
import sys
import uuid
import tempfile
import sqlite3
import io
import re
import contextlib

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

# Core Inference Clients
qwen_client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=hf_token)
whisper_client = InferenceClient(model="openai/whisper-large-v3", token=hf_token)

DB_PATH = "thunder_memory.db"

# --- PERSISTENT SQL DATABASE ENGINE ---
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
    conn.execute(
        "INSERT INTO history (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

def clear_session_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def load_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT role, content FROM history WHERE session_id = ? ORDER BY id",
        (session_id,)
    ).fetchall()
    conn.close()
    return [{"role": role, "content": content} for role, content in rows]

init_db()

DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder AI, an elite, high-velocity co-founder and AI collaborator. "
    "You provide hyper-accurate, structured, and witty responses using crisp headings and bullet points. "
    "When generating code, always wrap runnable Python blocks in standard markdown ```python blocks."
)

# --- AUTOMATIC CODE EXTRACTION UTILITY ---
def extract_python_code(text):
    """Extracts the first or most recent python block from response text."""
    matches = re.findall(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return None

# --- ADVANCED TOOLS & EXECUTION ---
def run_python_interpreter(code_str):
    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer), contextlib.redirect_stderr(output_buffer):
            exec_globals = {"__builtins__": __builtins__}
            exec(code_str, exec_globals)
        res = output_buffer.getvalue()
        return res if res.strip() else "[Code Executed Successfully - No Stdout Output]"
    except Exception as e:
        return f"Python Execution Error: {e}"

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

def web_search(query, max_results=3):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results: return ""
        return "\n".join(f"- {r.get('title','')}: {r.get('body','')} ({r.get('href', '')})" for r in results[:max_results])
    except Exception as e:
        return f"[Research Error: {e}]"

# --- MULTI-MODEL ROUTER ---
def stream_model_response(model_choice, messages, temp, tokens):
    if model_choice == "Gemini 1.5 Flash":
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            yield "⚠️ **Gemini Error:** `GEMINI_API_KEY` is missing in Render environment variables."
            return
        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            system_instr = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
            contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages if m["role"] != "system"]

            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=contents,
                config={"system_instruction": system_instr, "temperature": float(temp), "max_output_tokens": int(tokens)}
            )
            yield response.text or "[Empty Response]"
            return
        except Exception as e:
            yield f"⚠️ **Gemini API Error:** {e}"
            return

    elif model_choice == "OpenAI (GPT-4o)":
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            yield "⚠️ **OpenAI Error:** `OPENAI_API_KEY` environment variable is missing on Render."
            return
        try:
            from openai import OpenAI
            oai_client = OpenAI(api_key=openai_key)
            response_text = ""
            stream = oai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=float(temp),
                max_tokens=int(tokens),
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
                    yield response_text
            return
        except Exception as e:
            yield f"⚠️ **OpenAI API Error:** {e}"
            return

    elif model_choice == "Claude 3.5 Sonnet":
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
        if not anthropic_key:
            yield "⚠️ **Anthropic Error:** `ANTHROPIC_API_KEY` environment variable is missing on Render."
            return
        try:
            import anthropic
            ant_client = anthropic.Anthropic(api_key=anthropic_key)
            system_instr = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
            anthropic_msgs = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]

            response_text = ""
            with ant_client.messages.stream(
                model="claude-3-5-sonnet-20241022",
                max_tokens=int(tokens),
                temperature=float(temp),
                system=system_instr,
                messages=anthropic_msgs
            ) as stream:
                for text_chunk in stream.text_stream:
                    response_text += text_chunk
                    yield response_text
            return
        except Exception as e:
            yield f"⚠️ **Claude API Error:** {e}"
            return

    elif model_choice == "Perplexity (Sonar)":
        pplx_key = os.environ.get("PERPLEXITY_API_KEY")
        if not pplx_key:
            yield "⚠️ **Perplexity Error:** `PERPLEXITY_API_KEY` environment variable is missing on Render."
            return
        try:
            from openai import OpenAI
            pplx_client = OpenAI(api_key=pplx_key, base_url="[https://api.perplexity.ai](https://api.perplexity.ai)")
            response_text = ""
            stream = pplx_client.chat.completions.create(
                model="sonar-pro",
                messages=messages,
                temperature=float(temp),
                max_tokens=int(tokens),
                stream=True
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
                    yield response_text
            return
        except Exception as e:
            yield f"⚠️ **Perplexity API Error:** {e}"
            return

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

    else:
        full_res = ""
        for token in qwen_client.chat_completion(messages, max_tokens=int(tokens), temperature=float(temp), stream=True):
            chunk = token.choices[0].delta.content
            if chunk:
                full_res += chunk
                yield full_res

# --- CUSTOM THEME MATRIX ---
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
        with gr.Column(scale=4, elem_classes=["panel-card"]):
            session_selector = gr.Dropdown(choices=[], label="Workspace Session", interactive=True)
            new_session_btn = gr.Button("➕ New Workspace", size="sm")
            
            model_choice = gr.Dropdown(
                choices=[
                    "Qwen 2.5 7B (Default)",
                    "Gemini 1.5 Flash",
                    "OpenAI (GPT-4o)",
                    "Claude 3.5 Sonnet",
                    "Perplexity (Sonar)",
                    "DeepSeek R1 (Reasoning)"
                ],
                value="Qwen 2.5 7B (Default)",
                label="Select AI Model Engine"
            )
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Instructions", lines=2)
            
            with gr.Row():
                temperature = gr.Slider(0.1, 1.5, value=0.70, step=0.05, label="Temperature")
                max_tokens = gr.Slider(256, 4096, value=1536, step=128, label="Max Tokens")

            file_input = gr.File(label="Attach File / Document", file_count="single")
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice Input")
            research_toggle = gr.Checkbox(label="🔍 Deep Web Research Agent", value=False)
            clear_btn = gr.Button("Reset Chat", variant="stop")

        with gr.Column(scale=8):
            chatbot = gr.Chatbot(height=520, elem_classes=["chatbot-container"], type="messages")
            with gr.Row():
                msg = gr.Textbox(placeholder="Message Thunder AI...", show_label=False, container=False, scale=9)
                send_btn = gr.Button("⚡", variant="primary", scale=1)

        with gr.Column(scale=6, elem_classes=["panel-card"]):
            gr.Markdown("### 🛠️ Interactive Artifacts & Execution")
            artifact_code = gr.Code(label="Python Code Sandbox", language="python", lines=12)
            exec_btn = gr.Button("▶️ Execute Code", variant="primary", size="sm")
            exec_output = gr.Textbox(label="Execution Output", lines=6, interactive=False)

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

    def bot_reply(history, sys_prompt, model_sel, temp, tokens, f_context, research_on, sid, current_code):
        if not history: yield history, history, current_code; return

        last_user = history[-1]["content"]
        search_context = web_search(last_user) if research_on else ""

        messages = [{"role": "system", "content": sys_prompt}]
        if search_context: messages.append({"role": "system", "content": f"Web Research:\n{search_context}"})
        if f_context: messages.append({"role": "system", "content": f"Document Context:\n{f_context}"})

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
        [chat_state, system_prompt, model_choice, temperature, max_tokens, file_context_state, research_toggle, session_id, artifact_code], 
        [chatbot, chat_state, artifact_code]
    )

    send_btn.click(user_send, [msg, chat_state, session_id], [msg, chatbot, chat_state]).then(
        bot_reply, 
        [chat_state, system_prompt, model_choice, temperature, max_tokens, file_context_state, research_toggle, session_id, artifact_code], 
        [chatbot, chat_state, artifact_code]
    )

    exec_btn.click(run_python_interpreter, inputs=[artifact_code], outputs=[exec_output])
    clear_btn.click(lambda sid: (clear_session_history(sid), [], [])[1:], inputs=[session_id], outputs=[chatbot, chat_state])

port_number = int(os.environ.get("PORT", 10000))
demo.queue(default_concurrency_limit=4).launch(server_name="0.0.0.0", server_port=port_number)
        
