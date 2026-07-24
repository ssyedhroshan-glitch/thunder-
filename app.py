import os
import uuid
import tempfile
import sqlite3
import threading
import gradio as gr

from huggingface_hub import InferenceClient

# Optional API integrations with safe import wrappers
try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Securely retrieve API tokens
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")

# Primary High-Velocity Inference Clients
qwen_client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)
whisper_client = InferenceClient("openai/whisper-large-v3", token=HF_TOKEN)
tts_client = InferenceClient("microsoft/speecht5_tts", token=HF_TOKEN)

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if (HAS_GENAI and GEMINI_API_KEY) else None
claude_client = Anthropic(api_key=CLAUDE_API_KEY) if (HAS_ANTHROPIC and CLAUDE_API_KEY) else None

MAX_CONTEXT_TURNS = 12
DB_PATH = "thunder_v30_memory.db"

# --- SESSION MEMORY (SQLite WAL Mode with Thread Safety) ---
_db_lock = threading.Lock()
_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.execute("PRAGMA journal_mode=WAL")
_conn.execute("PRAGMA synchronous=NORMAL")

def init_db():
    with _db_lock:
        _conn.execute("""
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON history(session_id)")
        _conn.commit()

def load_history_dict(session_id):
    with _db_lock:
        rows = _conn.execute(
            "SELECT role, content FROM history WHERE session_id = ? ORDER BY id ASC",
            (session_id,)
        ).fetchall()
    return [{"role": role, "content": content} for role, content in rows]

def save_message(session_id, role, content):
    with _db_lock:
        _conn.execute(
            "INSERT INTO history (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        _conn.commit()

def clear_history(session_id):
    with _db_lock:
        _conn.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
        _conn.commit()

init_db()

DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder v30.0, an elite, highly intelligent AI collaborator and engineer. "
    "Provide clear, crisp, and insightful responses with strong technical accuracy. "
    "Format complex information into well-structured markdown with bold subheadings and bullet points. "
    "Maintain a supportive, authentic, and direct peer voice."
)

# --- UTILITIES & TOOL PIPELINES ---
def transcribe(audio_path):
    if not audio_path:
        return ""
    try:
        result = whisper_client.automatic_speech_recognition(audio_path)
        return result.text if hasattr(result, "text") else str(result)
    except Exception as e:
        return f"[Transcription Error: {str(e)}]"

def speak(text):
    if not text:
        return None
    try:
        clean_text = text.replace("*", "").replace("#", "").replace("`", "").replace("- ", "")[:250]
        audio_bytes = tts_client.text_to_speech(clean_text)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name
    except Exception:
        return None

def read_file(file_obj):
    if file_obj is None:
        return ""
    try:
        file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = "\n".join((page.extract_text() or "") for i, page in enumerate(reader.pages) if i < 20)
        elif ext in (".txt", ".md", ".csv", ".py", ".json", ".js", ".ts", ".html", ".css"):
            with open(file_path, "r", errors="ignore", encoding="utf-8") as f:
                text = f.read(20000)
        else:
            return f"[Attached File: {os.path.basename(file_path)}]"
        return text[:15000]
    except Exception as e:
        return f"[Parsing Error: {e}]"

def web_search(query, max_results=3):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        return "\n".join(f"- {r.get('title','')}: {r.get('body','')} ({r.get('href', '')})" for r in results[:max_results])
    except Exception as e:
        return f"[Search Error: {e}]"

# --- ENGINE SELECTION & INFERENCE ROUTING ---
def choose_model(message, forced_engine):
    if forced_engine != "Auto-Route":
        return forced_engine
    
    text = (message or "").lower().strip()
    if any(k in text for k in ["code", "refactor", "bug", "write", "essay", "architecture", "design"]):
        return "Claude 3.5 Sonnet" if claude_client else "Qwen 2.5 7B"
    if any(k in text for k in ["what is", "explain", "summarize", "search", "who is", "math"]):
        return "Gemini 1.5 Flash" if gemini_client else "Qwen 2.5 7B"
    
    return "Qwen 2.5 7B"

def query_llm(engine, messages, system_prompt, temperature, max_tokens):
    # 1. Gemini Engine Path
    if engine == "Gemini 1.5 Flash" and gemini_client:
        try:
            contents = []
            for m in messages:
                role = "user" if m["role"] == "user" else "model"
                contents.append({"role": role, "parts": [m["content"]]})
            
            response = gemini_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=float(temperature),
                    max_output_tokens=int(max_tokens),
                )
            )
            return response.text, "Gemini 1.5 Flash"
        except Exception:
            pass  # Fallback to Qwen

    # 2. Claude Engine Path
    if engine == "Claude 3.5 Sonnet" and claude_client:
        try:
            formatted_msgs = [{"role": m["role"], "content": m["content"]} for m in messages]
            response = claude_client.messages.create(
                model="claude-3-5-sonnet-latest",
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                system=system_prompt,
                messages=formatted_msgs,
            )
            return response.content[0].text, "Claude 3.5 Sonnet"
        except Exception:
            pass  # Fallback to Qwen

    # 3. Native Free Hugging Face Qwen Fallback
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = qwen_client.chat.completions.create(
            messages=full_messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature)
        )
        return response.choices[0].message.content, "Qwen 2.5 7B (HF Core)"
    except Exception as e:
        return f"[Engine Failure: {str(e)}]", "Error"

# --- CYBER MATRIX GRAPHICS & UI STYLING ---
custom_css = """
footer {visibility: hidden;}
body, .gradio-container {
    background: #050811 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.header-box {
    background: linear-gradient(135deg, rgba(15,23,42,0.9) 0%, rgba(30,41,59,0.5) 100%);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.header-title {
    background: linear-gradient(90deg, #38bdf8, #a855f7, #ec4899);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-size: 2rem;
    font-weight: 800;
    margin: 0;
}
.panel-card {
    background-color: #0b1329 !important;
    border: 1px solid #1e293b !important;
    border-radius: 14px !important;
    padding: 14px !important;
}
.chatbot-container {
    background-color: #070d1d !important;
    border: 1px solid rgba(56, 189, 248, 0.2) !important;
    border-radius: 16px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.45);
}
.action-btn {
    border-radius: 10px !important;
    font-weight: 600 !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="violet"), css=custom_css) as demo:
    session_id = gr.State(None)
    chat_state = gr.State([])
    file_context_state = gr.State("")

    # TOP CONTROL BAR
    with gr.Column(elem_classes=["header-box"]):
        with gr.Row():
            with gr.Column(scale=8):
                gr.Markdown("<h1 class='header-title'>⚡ THUNDER WORKSPACE v30.0</h1>")
                gr.Markdown("<p style='color: #94a3b8; margin: 0;'>Hyper-Engine AI Workspace • Adaptive Multi-Model Core with Persistent Memory</p>")
            with gr.Column(scale=4, min_width=220):
                engine_select = gr.Dropdown(
                    choices=["Auto-Route", "Qwen 2.5 7B", "Gemini 1.5 Flash", "Claude 3.5 Sonnet"],
                    value="Auto-Route",
                    label="🧠 Active Model Pipeline",
                    container=True
                )

    # MAIN CONTENT GRID
    with gr.Row():
        # LEFT CONTROL PANEL
        with gr.Column(scale=3, elem_classes=["panel-card"]):
            gr.Markdown("### 🛠 Tools & Context")
            research_toggle = gr.Checkbox(label="🔍 Enable Web Search", value=False)
            autoplay_audio = gr.Checkbox(label="🔊 Auto-Play Speech", value=False)
            
            with gr.Accordion("📎 Attach File / Voice", open=True):
                file_input = gr.File(label="Upload Document", file_count="single", file_types=[".txt", ".pdf", ".md", ".py", ".json", ".csv"])
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice Dictation")

            with gr.Accordion("⚙️ Model Parameters", open=False):
                system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Instructions", lines=3)
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature")
                max_tokens = gr.Slider(256, 4096, value=2048, step=128, label="Max Output Tokens")

            clear_btn = gr.Button("🗑️ Clear Workspace", variant="stop", elem_classes=["action-btn"])
            engine_status = gr.Markdown("<small>Engine: Standby</small>")

        # RIGHT CHAT INTERFACE
        with gr.Column(scale=7):
            chatbot = gr.Chatbot(
                height=540,
                elem_classes=["chatbot-container"],
                type="messages",
                show_copy_button=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="Type your message or dictate audio...",
                    container=False,
                    scale=8
                )
                send_btn = gr.Button("⚡ Send", variant="primary", scale=2, elem_classes=["action-btn"])

    reply_audio = gr.Audio(autoplay=True, visible=False)

    # --- SESSION INITIALIZATION ---
    def start_session():
        new_id = str(uuid.uuid4())
        initial_history = load_history_dict(new_id)
        return new_id, initial_history, initial_history

    demo.load(start_session, None, [session_id, chatbot, chat_state])

    # EVENT HANDLERS
    audio_input.change(lambda a: transcribe(a), inputs=[audio_input], outputs=[msg])
    file_input.change(lambda f: read_file(f), inputs=[file_input], outputs=[file_context_state])

    def user_send(message, f_context, history, sid):
        message = (message or "").strip()
        if not message:
            if f_context:
                message = "[Document Context Loaded for Reference]"
            else:
                return "", history, history
        
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": ""}
        ]
        save_message(sid, "user", message)
        return "", updated_history, updated_history

    def bot_reply(history, sys_prompt, temp, tokens, f_context, research_on, engine_choice, sid):
        if not history:
            yield history, history, "Engine: Standby"
            return

        last_user = history[-2]["content"]
        
        # Build Web Search Context
        search_context = web_search(last_user) if research_on else ""
        
        # Construct Prompt Context
        combined_sys = sys_prompt
        if search_context:
            combined_sys += "\n\nLive Search Context:\n" + search_context
        if f_context:
            combined_sys += "\n\nUploaded Workspace Reference:\n" + f_context

        # Build Context Message Window
        recent_messages = history[:-2][-MAX_CONTEXT_TURNS*2:]
        payload = [{"role": m["role"], "content": m["content"]} for m in recent_messages]
        payload.append({"role": "user", "content": last_user})

        # Query Target Engine
        target_engine = choose_model(last_user, engine_choice)
        response_text, active_engine = query_llm(target_engine, payload, combined_sys, temp, tokens)

        history[-1]["content"] = response_text
        save_message(sid, "assistant", response_text)

        yield history, history, f"Active Model: **{active_engine}**"

    def bot_speak(history, audio_enabled):
        if not history or not audio_enabled:
            return None
        text = history[-1]["content"]
        return speak(text) if text else None

    def reset_media():
        return None, None, ""

    def do_clear(sid):
        clear_history(sid)
        return [], [], "<small>Engine: Workspace Cleared</small>"

    # Event Wireup
    msg.submit(
        user_send, [msg, file_context_state, chat_state, session_id], [msg, chatbot, chat_state]
    ).then(
        bot_reply, [chat_state, system_prompt, temperature, max_tokens, file_context_state, research_toggle, engine_select, session_id], [chatbot, chat_state, engine_status]
    ).then(
        bot_speak, [chat_state, autoplay_audio], [reply_audio]
    ).then(
        reset_media, None, [file_input, audio_input, file_context_state]
    )

    send_btn.click(
        user_send, [msg, file_context_state, chat_state, session_id], [msg, chatbot, chat_state]
    ).then(
        bot_reply, [chat_state, system_prompt, temperature, max_tokens, file_context_state, research_toggle, engine_select, session_id], [chatbot, chat_state, engine_status]
    ).then(
        bot_speak, [chat_state, autoplay_audio], [reply_audio]
    ).then(
        reset_media, None, [file_input, audio_input, file_context_state]
    )

    clear_btn.click(do_clear, [session_id], [chatbot, chat_state, engine_status])

port_number = int(os.environ.get("PORT", 10000))
demo.queue(default_concurrency_limit=8).launch(server_name="0.0.0.0", server_port=port_number)
