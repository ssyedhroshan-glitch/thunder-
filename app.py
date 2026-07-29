import os
import uuid
import tempfile
import sqlite3
import threading
import gradio as gr

from huggingface_hub import InferenceClient

# Safe imports for external API providers
HAS_GENAI = False
genai_types = None
try:
    from google import genai
    from google.genai import types as g_types
    genai_types = g_types
    HAS_GENAI = True
except Exception:
    HAS_GENAI = False

HAS_ANTHROPIC = False
try:
    from anthropic import Anthropic
    HAS_ANTHROPIC = True
except Exception:
    HAS_ANTHROPIC = False

# Retrieve API tokens safely
HF_TOKEN = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")

# Primary Inference Clients
hf_client = InferenceClient(token=HF_TOKEN) if HF_TOKEN else InferenceClient()
whisper_client = InferenceClient("openai/whisper-large-v3", token=HF_TOKEN) if HF_TOKEN else None
tts_client = InferenceClient("microsoft/speecht5_tts", token=HF_TOKEN) if HF_TOKEN else None

gemini_client = genai.Client(api_key=GEMINI_API_KEY) if (HAS_GENAI and GEMINI_API_KEY) else None
claude_client = Anthropic(api_key=CLAUDE_API_KEY) if (HAS_ANTHROPIC and CLAUDE_API_KEY) else None

MAX_CONTEXT_TURNS = 12
DB_PATH = "thunder_v31_memory.db"

# --- PERSISTENT SESSION MEMORY ---
_db_lock = threading.Lock()
try:
    _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    _conn.execute("PRAGMA journal_mode=WAL")
    _conn.execute("PRAGMA synchronous=NORMAL")
except Exception:
    _conn = sqlite3.connect(":memory:", check_same_thread=False)

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

try:
    init_db()
except Exception:
    pass

DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder v31.0, an elite, highly intelligent AI collaborator. "
    "Provide clear, crisp, and insightful responses. "
    "Use bolding for emphasis, bullet points for lists, and code blocks for technical syntax."
)

# --- UTILITIES ---
def transcribe(audio_path):
    if not audio_path or not whisper_client:
        return ""
    try:
        result = whisper_client.automatic_speech_recognition(audio_path)
        return result.text if hasattr(result, "text") else str(result)
    except Exception:
        return "[Audio transcription failed]"

def speak(text):
    if not text or not tts_client:
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
        return "", "No file loaded."
    try:
        file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
        filename = os.path.basename(file_path)
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = "\n".join((page.extract_text() or "") for i, page in enumerate(reader.pages) if i < 20)
        else:
            with open(file_path, "r", errors="ignore", encoding="utf-8") as f:
                text = f.read(20000)
        return text[:15000], f"✅ **Loaded:** {filename}"
    except Exception as e:
        return "", f"❌ **Error parsing file:** {str(e)}"

def web_search(query, max_results=3):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        return "\n".join(f"Title: {r.get('title','')}\nSnippet: {r.get('body','')}\nURL: {r.get('href', '')}\n---" for r in results[:max_results])
    except Exception:
        return ""

def choose_model(message, forced_engine):
    if forced_engine != "Auto-Route":
        return forced_engine
    text = (message or "").lower().strip()
    if any(k in text for k in ["code", "refactor", "bug", "write", "essay", "architecture"]):
        return "Claude 3.5 Sonnet" if claude_client else "Qwen 2.5 7B"
    if any(k in text for k in ["what is", "explain", "summarize", "search", "who is", "math"]):
        return "Gemini 1.5 Flash" if gemini_client else "Qwen 2.5 7B"
    return "Qwen 2.5 7B"

def query_llm(engine, messages, system_prompt, temperature, max_tokens):
    # Gemini Execution
    if engine == "Gemini 1.5 Flash":
        if not gemini_client:
            return "⚠️ **System Error:** Gemini API Key is missing or invalid.", "Error"
        try:
            contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]} for m in messages]
            response = gemini_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=contents,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=float(temperature),
                    max_output_tokens=int(max_tokens),
                )
            )
            return response.text, "Gemini 1.5 Flash"
        except Exception as e:
            return f"⚠️ **Gemini API Error:** {str(e)}", "Error"

    # Claude Execution
    if engine == "Claude 3.5 Sonnet":
        if not claude_client:
            return "⚠️ **System Error:** Anthropic API Key is missing or invalid.", "Error"
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
        except Exception as e:
            return f"⚠️ **Claude API Error:** {str(e)}", "Error"

    # Qwen (HuggingFace Core) Execution
    try:
        full_messages = [{"role": "system", "content": system_prompt}] + messages
        response = hf_client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=full_messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature)
        )
        return response.choices[0].message.content, "Qwen 2.5 7B (HF Core)"
    except Exception as e:
        return f"⚠️ **HuggingFace API Error:** Endpoint may be busy. {str(e)}", "Error"

# --- CUSTOM UI THEME ---
custom_css = """
#main-container { max-width: 1400px; margin: 0 auto; padding-top: 20px; }
.gradio-container { font-family: 'Inter', sans-serif; }
.chatbot-bubble { border-radius: 12px; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"), css=custom_css, title="Thunder v31.0") as demo:
    session_id = gr.State(None)
    chat_state = gr.State([])
    file_context_state = gr.State("")

    with gr.Column(elem_id="main-container"):
        with gr.Row():
            gr.Markdown("# ⚡ THUNDER WORKSPACE \n<small>v31.0 Pro Edition</small>", scale=8)
            engine_status = gr.Markdown("**Status:** 🟢 Ready | **Engine:** Standby", scale=4)

        with gr.Row():
            # --- LEFT SIDEBAR: TOOLS ---
            with gr.Column(scale=3):
                engine_select = gr.Dropdown(
                    choices=["Auto-Route", "Qwen 2.5 7B", "Gemini 1.5 Flash", "Claude 3.5 Sonnet"],
                    value="Auto-Route",
                    label="🧠 Routing Engine",
                    interactive=True
                )
                
                with gr.Group():
                    gr.Markdown("### 🛠 Integrations")
                    research_toggle = gr.Checkbox(label="🔍 Live Web Search", value=False)
                    autoplay_audio = gr.Checkbox(label="🔊 Auto-Play Voice Replies", value=False)
                
                with gr.Accordion("📎 Workspace Documents", open=True):
                    file_input = gr.File(
                        label="Drop a PDF, TXT, or Code file", 
                        file_count="single", 
                        file_types=[".txt", ".pdf", ".md", ".py", ".json", ".csv"]
                    )
                    file_status = gr.Markdown("<small>No file loaded.</small>")
                    audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice Dictation")

                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="Persona Prompt", lines=4)
                    temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Creativity (Temp)")
                    max_tokens = gr.Slider(256, 4096, value=2048, step=128, label="Response Length")

                clear_btn = gr.Button("🗑️ Clear Session Memory", variant="stop")

            # --- RIGHT PANEL: CHAT ---
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    height=600, 
                    type="messages", 
                    show_copy_button=True, 
                    avatar_images=(None, "https://cdn-icons-png.flaticon.com/512/6183/6183917.png")
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        show_label=False, 
                        placeholder="Type a message or use the microphone...", 
                        scale=8,
                        container=False
                    )
                    send_btn = gr.Button("⚡ Send", variant="primary", scale=1)

    reply_audio = gr.Audio(autoplay=True, visible=False)

    # --- EVENT HANDLERS ---
    def start_session():
        new_id = str(uuid.uuid4())
        initial_history = load_history_dict(new_id)
        return new_id, initial_history, initial_history

    demo.load(start_session, None, [session_id, chatbot, chat_state])

    audio_input.change(lambda a: transcribe(a), inputs=[audio_input], outputs=[msg])
    
    def handle_file_upload(f):
        text, status_msg = read_file(f)
        return text, status_msg
    
    file_input.change(handle_file_upload, inputs=[file_input], outputs=[file_context_state, file_status])

    def user_send(message, f_context, history, sid):
        message = (message or "").strip()
        if not message:
            if f_context:
                message = "[Please analyze the attached document]"
            else:
                return "", history, history
        
        updated_history = list(history) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": ""}
        ]
        save_message(sid, "user", message)
        return "", updated_history, updated_history

    def bot_reply(history, sys_prompt, temp, tokens, f_context, research_on, engine_choice, sid):
        if not history or len(history) < 2:
            yield history, history, "**Status:** 🟢 Ready | **Engine:** Standby"
            return

        yield history, history, "**Status:** ⏳ Processing..."

        last_user = history[-2]["content"]
        search_context = web_search(last_user) if research_on else ""
        
        combined_sys = sys_prompt
        if search_context:
            combined_sys += f"\n\n<web_search_results>\n{search_context}\n</web_search_results>"
        if f_context:
            combined_sys += f"\n\n<document_context>\n{f_context}\n</document_context>\nInstruct the model to use the <document_context> to answer user queries."

        recent_messages = history[:-2][-MAX_CONTEXT_TURNS*2:]
        payload = [{"role": m["role"], "content": m["content"]} for m in recent_messages]
        payload.append({"role": "user", "content": last_user})

        target_engine = choose_model(last_user, engine_choice)
        response_text, active_engine = query_llm(target_engine, payload, combined_sys, temp, tokens)

        history[-1]["content"] = response_text
        save_message(sid, "assistant", response_text)

        yield history, history, f"**Status:** 🟢 Ready | **Engine:** {active_engine}"

    def bot_speak(history, audio_enabled):
        if not history or not audio_enabled:
            return None
        text = history[-1]["content"]
        return speak(text) if text else None

    def reset_media():
        return None, None, "", "<small>No file loaded.</small>"

    def do_clear(sid):
        clear_history(sid)
        return [], [], "**Status:** 🟢 Session Cleared | **Engine:** Standby"

    # Submission routing
    msg.submit(
        user_send, [msg, file_context_state, chat_state, session_id], [msg, chatbot, chat_state]
    ).then(
        bot_reply, [chat_state, system_prompt, temperature, max_tokens, file_context_state, research_toggle, engine_select, session_id], [chatbot, chat_state, engine_status]
    ).then(
        bot_speak, [chat_state, autoplay_audio], [reply_audio]
    ).then(
        reset_media, None, [file_input, audio_input, file_context_state, file_status]
    )

    send_btn.click(
        user_send, [msg, file_context_state, chat_state, session_id], [msg, chatbot, chat_state]
    ).then(
        bot_reply, [chat_state, system_prompt, temperature, max_tokens, file_context_state, research_toggle, engine_select, session_id], [chatbot, chat_state, engine_status]
    ).then(
        bot_speak, [chat_state, autoplay_audio], [reply_audio]
    ).then(
        reset_media, None, [file_input, audio_input, file_context_state, file_status]
    )

    clear_btn.click(do_clear, [session_id], [chatbot, chat_state, engine_status])

# --- CRITICAL RENDER LAUNCH CONFIGURATION ---
port_number = int(os.environ.get("PORT", 10000))
demo.launch(
    server_name="0.0.0.0", 
    server_port=port_number,
    share=False,
    api_open=False,
    show_api=False
            )
        
