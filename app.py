import os
import uuid
import tempfile
import sqlite3
import gradio as gr
from huggingface_hub import InferenceClient

hf_token = os.environ.get("HF_TOKEN")

# High-Velocity Inference Clients
client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=hf_token)
whisper_client = InferenceClient(model="openai/whisper-large-v3", token=hf_token)
tts_client = InferenceClient(model="microsoft/speecht5_tts", token=hf_token)

DB_PATH = "thunder_memory.db"

# --- CORE PERSISTENT MULTI-SESSION SQL DATABASE ENGINE ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    # Create sessions table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Create history table
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
    
    # Initialize a default session if no sessions exist
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

def rename_session_in_db(session_id, new_name):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE sessions SET name = ? WHERE id = ?", (new_name, session_id))
    conn.commit()
    conn.close()

def delete_session_from_db(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
    
    # Secure fallback fallback: Ensure at least one session remains
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sessions")
    if cursor.fetchone()[0] == 0:
        default_id = str(uuid.uuid4())
        cursor.execute("INSERT INTO sessions (id, name) VALUES (?, ?)", (default_id, "Default Workspace"))
    conn.commit()
    conn.close()

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
    
    # Universal fallback list parser
    chat_list = []
    for role, content in rows:
        chat_list.append({"role": role, "content": content})
    return chat_list

init_db()

DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder, an elite, tech-savvy AI collaborator with a sharp mind and a touch of dry wit. "
    "You talk to the user as a brilliant, supportive peer and co-founder. "
    "Provide highly insightful, direct, and scannable answers using short headings and clean bullet points. "
    "Keep your tone authentic, grounded, and engaging. Never use robotic disclaimers."
)

# --- BACKEND MULTIMEDIA WORKFLOW PIPELINES ---
def transcribe(audio_path):
    if not audio_path:
        return ""
    try:
        result = whisper_client.automatic_speech_recognition(audio_path)
        return result.text if hasattr(result, "text") else str(result)
    except Exception as e:
        return f"[Transcription Error: {e}]"

def speak(text, enabled=True):
    if not text or not enabled:
        return None
    try:
        clean_text = text.replace("*", "").replace("#", "").replace("`", "")[:200]
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
            text = "\n".join((page.extract_text() or "") for i, page in enumerate(reader.pages) if i < 15) # Safe page count limits
        elif ext in (".txt", ".md", ".csv", ".py", ".json", ".js", ".ts"):
            with open(file_path, "r", errors="ignore", encoding="utf-8") as f:
                text = f.read(15000) # Prevents memory leaks with smart context clipping
        else:
            return f"[Attached Asset File: {os.path.basename(file_path)}]"
        return text[:12000]
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
        return f"[Deep Research Error: {e}]"

# --- CONTEXT-BUDGET OPTIMIZED PROMPT ASSEMBLY ---
def build_messages(chat_history, system_prompt, file_context, search_context):
    messages = [{"role": "system", "content": system_prompt}]
    if search_context:
        messages.append({"role": "system", "content": "Deep Research Results:\n\n" + search_context})
    if file_context:
        messages.append({"role": "system", "content": "Attached Document Reference:\n\n" + file_context})
    
    # Smart context trimming: Feed only the last 10 messages to avoid 504 timeouts
    recent_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
    
    for msg in recent_history:
        if isinstance(msg, dict):
            messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        elif isinstance(msg, (list, tuple)) and len(msg) >= 2:
            if msg[0]: messages.append({"role": "user", "content": msg[0]})
            if msg[1]: messages.append({"role": "assistant", "content": msg[1]})
    return messages

def stream_reply(messages, temperature, max_tokens):
    text = ""
    for chunk in client.chat.completions.create(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True
    ):
        text += chunk.choices[0].delta.content or ""
        yield text

# --- PREMIUM CYBERPUNK THEME DESIGN MATRIX ---
custom_css = """
footer {visibility: hidden;}
body, .gradio-container {background-color: #0b0f19 !important;}
.panel-card {
    background-color: #121826 !important; 
    border: 1px solid #1f293d !important;
    border-radius: 12px !important;
    padding: 14px !important;
    margin-bottom: 10px !important;
}
.chatbot-container {
    border: 1px solid #1f293d !important;
    border-radius: 12px !important;
}
.header-row {
    align-items: center !important;
    margin-bottom: 10px;
}
.flex-end-layout {
    display: flex !important;
    justify-content: flex-end !important;
    gap: 10px !important;
    align-items: center !important;
}
.console-row {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 8px !important;
    width: 100% !important;
}
.msg-container {
    flex-grow: 1 !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"), 
    css=custom_css,
    js="() => { document.querySelector('body').classList.add('dark'); }"
) as demo:
    
    session_id = gr.State(None)
    chat_state = gr.State([])
    file_context_state = gr.State("")
    features_visible = gr.State(False)

    # TOP ALIGNED DYNAMIC CONFIGURATION HEADER
    with gr.Row(elem_classes=["header-row"]):
        with gr.Column(scale=3):
            settings_toggle = gr.Checkbox(label="⚙️ Settings Option", value=False, container=False)
        with gr.Column(scale=4):
            gr.Markdown("<center><h2 style='margin:0; padding:0; color:#22d3ee;'>⚡ THUNDER WORKSPACE</h2></center>")
        with gr.Column(scale=5, elem_classes=["flex-end-layout"]):
            theme_choice = gr.Radio(["Dark Matrix", "Light Slate"], value="Dark Matrix", show_label=False, container=False)
            session_selector = gr.Dropdown(choices=[], label="Workspace Session", interactive=True, container=False)
            new_session_btn = gr.Button("➕ Session", size="sm", variant="secondary")
            clear_btn = gr.Button("🆕 Reset", variant="stop", size="sm")

    # EXPANDABLE SYSTEM PREFERENCE SYSTEM DRAWER
    with gr.Group(visible=False, elem_classes=["panel-card"]) as settings_panel:
        with gr.Row():
            with gr.Column(scale=6):
                system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="Prompt Core Constraints", lines=2)
            with gr.Column(scale=3):
                temperature = gr.Slider(0.1, 1.5, value=0.70, step=0.05, label="Temperature Matrix")
                max_tokens = gr.Slider(256, 4096, value=1536, step=128, label="Token Window")
            with gr.Column(scale=3):
                autoplay_audio = gr.Checkbox(label="🔊 Voice AutoPlay", value=True)
                session_rename_box = gr.Textbox(placeholder="Rename active workspace...", show_label=False)
                rename_session_btn = gr.Button("Rename Workspace", size="sm")

    chatbot = gr.Chatbot(height=480, elem_classes=["chatbot-container"])

    # HIDDEN EXPANDABLE OPTIONS VAULT
    with gr.Group(visible=False, elem_classes=["panel-card"]) as features_vault:
        gr.Markdown("🌟 **Additional Features Suite**")
        with gr.Row():
            with gr.Column(scale=2, min_width=120):
                file_input = gr.File(label="📎 Add File", file_count="single", file_types=[".txt", ".pdf", ".md", ".py", ".json", ".csv"])
            with gr.Column(scale=2, min_width=120):
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤 Mic")
            with gr.Column(scale=2, min_width=120):
                research_toggle = gr.Checkbox(label="🔍 Deep Research", value=False)
            with gr.Column(scale=2, min_width=120):
                camera_input = gr.Image(sources=["webcam"], type="filepath", label="📷 Camera")
            with gr.Column(scale=2, min_width=120):
                canvas_input = gr.Image(sources=["upload"], type="filepath", label="🎨 Image Editing")

    # NESTED CONSOLE MATRIX LAYOUT BAR
    with gr.Row(elem_classes=["console-row"]):
        features_btn = gr.Button("➕", variant="secondary", size="sm", min_width=50)
        with gr.Column(elem_classes=["msg-container"]):
            msg = gr.Textbox(
                show_label=False,
                placeholder="Message Thunder or expand options panel using '+'...",
                container=False
            )
        send_btn = gr.Button("⚡", variant="primary", size="sm", min_width=50)

    reply_audio = gr.Audio(autoplay=True, visible=False)

    # --- CONTROLLER INTERACTION LOGIC ---
    def start_session():
        init_db()
        sessions = get_all_sessions()
        # Default fallback to first active session ID
        active_id = sessions[0][0] if sessions else ""
        initial_history = load_history(active_id)
        
        # Build dropdown selection names
        session_choices = [(name, sid) for sid, name in sessions]
        return active_id, initial_history, initial_history, gr.update(choices=session_choices, value=active_id)

    demo.load(start_session, None, [session_id, chatbot, chat_state, session_selector])

    # Dynamic drawer mapping switches
    settings_toggle.change(lambda visible: gr.update(visible=visible), inputs=[settings_toggle], outputs=[settings_panel])
    
    def toggle_vault(current_state):
        return not current_state, gr.update(visible=not current_state)
    features_btn.click(toggle_vault, [features_visible], [features_visible, features_vault])

    # Dynamic Workspace Sessions Manager Events
    def switch_active_session(target_id):
        if not target_id:
            return gr.update(), [], []
        new_history = load_history(target_id)
        return target_id, new_history, new_history

    session_selector.change(switch_active_session, inputs=[session_selector], outputs=[session_id, chatbot, chat_state])

    def create_and_refresh_session():
        new_id = create_new_session("Workspace Session")
        sessions = get_all_sessions()
        session_choices = [(name, sid) for sid, name in sessions]
        new_history = load_history(new_id)
        return new_id, new_history, new_history, gr.update(choices=session_choices, value=new_id)

    new_session_btn.click(create_and_refresh_session, None, [session_id, chatbot, chat_state, session_selector])

    def rename_active_workspace(active_id, new_name):
        if not active_id or not new_name.strip():
            return gr.update()
        rename_session_in_db(active_id, new_name.strip())
        sessions = get_all_sessions()
        session_choices = [(name, sid) for sid, name in sessions]
        return gr.update(choices=session_choices, value=active_id)

    rename_session_btn.click(rename_active_workspace, [session_id, session_rename_box], [session_selector])

    theme_js = """
    (mode) => {
        const body = document.querySelector('body');
        if(mode === 'Light Slate') {
            body.style.backgroundColor = '#f7fafc';
            body.classList.remove('dark');
        } else {
            body.style.backgroundColor = '#0b0f19';
            body.classList.add('dark');
        }
    }
    """
    theme_choice.change(None, inputs=[theme_choice], js=theme_js)

    def on_audio(audio_path):
        if not audio_path: return gr.update()
        return transcribe(audio_path)
    audio_input.change(on_audio, inputs=[audio_input], outputs=[msg])

    def on_file(file_obj):
        return read_file(file_obj)
    file_input.change(on_file, inputs=[file_input], outputs=[file_context_state])

    def user_send(message, f_context, cam, sketch, history, sid):
        message = (message or "").strip()
        if not message:
            if f_context: message = "⚡ [File payload analyzed]"
            elif cam: message = "📷 [Camera stream frame captured]"
            elif sketch: message = "🎨 [Canvas frame updated]"
        if not message: return "", history, history
        
        if len(history) > 0 and isinstance(history[0], dict):
            new_history = history + [{"role": "user", "content": message}]
        else:
            new_history = history + [[message, ""]]
            
        save_message(sid, "user", message)
        return "", new_history, new_history

    def bot_reply(history, sys_prompt, temp, tokens, f_context, research_on, sid):
        if not history: return history, history
        
        is_dict = isinstance(history[0], dict)
        last_user = history[-1]["content"] if is_dict else history[-1][0]
        
        search_context = web_search(last_user) if research_on else ""
        messages = build_messages(history[:-1], sys_prompt, f_context, search_context)
        messages.append({"role": "user", "content": last_user})

        if is_dict:
            history = history + [{"role": "assistant", "content": ""}]
        else:
            history = history + [[last_user, ""]]

        final_text = ""
        try:
            for partial in stream_reply(messages, temp, tokens):
                final_text = partial
                if is_dict:
                    history[-1]["content"] = final_text
                else:
                    history[-1][1] = final_text
                yield history, history
            save_message(sid, "assistant", final_text)
        except Exception as e:
            err = f"Engine Error: {e}"
            if is_dict:
                history[-1]["content"] = err
            else:
                history[-1][1] = err
            save_message(sid, "assistant", err)
            yield history, history

    def bot_speak(history, audio_enabled):
        if not history or not audio_enabled: return None
        is_dict = isinstance(history[0], dict)
        text = history[-1]["content"] if is_dict else history[-1][1]
        if text: return speak(text, audio_enabled)
        return None

    def reset_media_slots():
        return None, None, None, ""

    def do_clear(sid):
        clear_session_history(sid)
        return [], [], ""

    msg.submit(
        user_send, [msg, file_context_state, camera_input, canvas_input, chat_state, session_id], [msg, chatbot, chat_state]
    ).then(
        bot_reply, [chat_state, system_prompt, temperature, max_tokens, file_context_state, research_toggle, session_id], [chatbot, chat_state]
    ).then(
        bot_speak, [chat_state, autoplay_audio], [reply_audio]
    ).then(
        reset_media_slots, None, [file_input, camera_input, canvas_input, file_context_state]
    )

    send_btn.click(
        user_send, [msg, file_context_state, camera_input, canvas_input, chat_state, session_id], [msg, chatbot, chat_state]
    ).then(
        bot_reply, [chat_state, system_prompt, temperature, max_tokens, file_context_state, research_toggle, session_id], [chatbot, chat_state]
    ).then(
        bot_speak, [chat_state, autoplay_audio], [reply_audio]
    ).then(
        reset_media_slots, None, [file_input, camera_input, canvas_input, file_context_state]
    )

    clear_btn.click(do_clear, [session_id], [chatbot, chat_state, file_context_state])

port_number = int(os.environ.get("PORT", 10000))
demo.queue(default_concurrency_limit=4).launch(server_name="0.0.0.0", server_port=port_number)
