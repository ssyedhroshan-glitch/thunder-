import os
import uuid
import tempfile
import sqlite3
import gradio as gr
from huggingface_hub import InferenceClient

hf_token = os.environ.get("HF_TOKEN")

# High-Velocity Inference Drivers
client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=hf_token)
whisper_client = InferenceClient(model="openai/whisper-large-v3", token=hf_token)
tts_client = InferenceClient(model="microsoft/speecht5_tts", token=hf_token)

DB_PATH = "thunder_memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON history(session_id)")
    conn.commit()
    conn.close()

def save_message(session_id, role, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO history (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
    conn.commit()
    conn.close()

def clear_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

def load_history(session_id):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT role, content FROM history WHERE session_id = ? ORDER BY id", (session_id,)).fetchall()
    conn.close()
    
    chat_tuples = []
    temp_user = None
    for role, content in rows:
        if role == "user":
            temp_user = content
        elif role == "assistant":
            if temp_user is not None:
                chat_tuples.append([temp_user, content])
                temp_user = None
            else:
                chat_tuples.append(["", content])
    if temp_user is not None:
        chat_tuples.append([temp_user, ""])
    return chat_tuples

init_db()

DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder, an elite, tech-savvy AI collaborator with a sharp mind and a touch of dry wit. "
    "You talk to the user as a brilliant, supportive peer and co-founder. "
    "Provide highly insightful, direct, and scannable answers using short headings and clean bullet points. "
    "Keep your tone authentic, grounded, and engaging."
)

def transcribe(audio_path):
    if not audio_path:
        return ""
    try:
        result = whisper_client.automatic_speech_recognition(audio_path)
        return result.text if hasattr(result, "text") else str(result)
    except Exception as e:
        return f"[Transcription Error: {e}]"

def speak(text):
    if not text:
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
            text = "\n".join((page.extract_text() or "") for i, page in enumerate(reader.pages) if i < 20)
        elif ext in (".txt", ".md", ".csv", ".py", ".json", ".js", ".ts"):
            with open(file_path, "r", errors="ignore", encoding="utf-8") as f:
                text = f.read(20000)
        else:
            return f"[Attached Asset File: {os.path.basename(file_path)}]"
        return text[:16000]
    except Exception as e:
        return f"[Parsing Error: {e}]"

def web_search(query, max_results=3):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        return "\n".join(f"- {r.get('title','')}: {r.get('body','')} ({r.get('href','')})" for r in results[:max_results])
    except Exception as e:
        return f"[Deep Research Error: {e}]"

def build_messages(chat_history, system_prompt, file_context, search_context):
    messages = [{"role": "system", "content": system_prompt}]
    if search_context:
        messages.append({"role": "system", "content": "Deep Research Results:\n\n" + search_context})
    if file_context:
        messages.append({"role": "system", "content": "Attached Document Reference:\n\n" + file_context})
    for u, a in chat_history:
        if u: messages.append({"role": "user", "content": u})
        if a: messages.append({"role": "assistant", "content": a})
    return messages

def stream_reply(messages, temperature, max_tokens):
    text = ""
    for chunk in client.chat.completions.create(messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True):
        text += chunk.choices[0].delta.content or ""
        yield text

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
.header-row {
    align-items: center !important;
    margin-bottom: 10px;
}
.flex-end-layout {
    display: flex !important;
    justify-content: flex-end !important;
    gap: 10px !important;
}
.console-row {
    align-items: center !important;
    gap: 8px !important;
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

    # TOP STRIP: Settings (Top Left) | Theme & Reset (Top Right)
    with gr.Row(elem_classes=["header-row"]):
        with gr.Column(scale=3):
            settings_toggle = gr.Checkbox(label="⚙️ Settings Option", value=False, container=False)
        with gr.Column(scale=4):
            gr.Markdown("<center><h2 style='margin:0; padding:0; color:#22d3ee;'>⚡ THUNDER WORKSPACE</h2></center>")
        with gr.Column(scale=3, elem_classes=["flex-end-layout"]):
            theme_choice = gr.Radio(["Dark Matrix", "Light Slate"], value="Dark Matrix", show_label=False, container=False)
            clear_btn = gr.Button("🆕 Reset", variant="stop", size="sm")

    # TOP LEFT EXPANDABLE SYSTEM DRAWER
    with gr.Group(visible=False, elem_classes=["panel-card"]) as settings_panel:
        with gr.Row():
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="Prompt Core Constraints", lines=2)
            temperature = gr.Slider(0.1, 1.5, value=0.70, step=0.05, label="Temperature Matrix")
            max_tokens = gr.Slider(256, 4096, value=1536, step=128, label="Token Window")

    chatbot = gr.Chatbot(height=480, elem_classes=["chatbot-container"])

    # HIDDEN EXPANDABLE ADDITIONAL FEATURES VAULT (Triggered by '+')
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
                canvas_input = gr.Image(sources=["upload"], tool="sketch", type="filepath", label="🎨 Image Editing")

    # CONSOLE BOX ROW: [+] on Left, Input in Center, [⚡] inside right corner
    with gr.Row(elem_classes=["console-row"]):
        with gr.Column(scale=1, min_width=50):
            features_btn = gr.Button("➕", variant="secondary")
        with gr.Column(scale=10):
            msg = gr.Textbox(
                show_label=False,
                placeholder="Message Thunder or expand options panel using '+'...",
                container=False
            )
        with gr.Column(scale=1, min_width=50):
            send_btn = gr.Button("⚡", variant="primary")

    reply_audio = gr.Audio(autoplay=True, visible=False)

    def start_session():
        sid = str(uuid.uuid4())
        return sid, load_history(sid), load_history(sid)

    demo.load(start_session, None, [session_id, chatbot, chat_state])

    # Toggle Handlers
    settings_toggle.change(lambda visible: gr.update(visible=visible), inputs=[settings_toggle], outputs=[settings_panel])
    
    def toggle_vault(current_state):
        return not current_state, gr.update(visible=not current_state)
    features_btn.click(toggle_vault, [features_visible], [features_visible, features_vault])

    # Theme Matrix Integration Switcher
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
            if f_context: message = "⚡ [Analyzed Document Payload Attached]"
            elif cam: message = "📷 [Webcam Frame Ingested]"
            elif sketch: message = "🎨 [Canvas Image Layout Modified]"
        if not message: return "", history, history
        
        history = history + [[message, ""]]
        save_message(sid, "user", message)
        return "", history, history

    def bot_reply(history, sys_prompt, temp, tokens, f_context, research_on, sid):
        if not history: return history, history
        last_user = history[-1][0]
        search_context = web_search(last_user) if research_on else ""
        
        messages = build_messages(history[:-1], sys_prompt, f_context, search_context)
        messages.append({"role": "user", "content": last_user})

        final_text = ""
        try:
            for partial in stream_reply(messages, temp, tokens):
                final_text = partial
                history[-1][1] = final_text
                yield history, history
            save_message(sid, "assistant", final_text)
        except Exception as e:
            err = f"Engine Error: {e}"
            history[-1][1] = err
            save_message(sid, "assistant", err)
            yield history, history

    def bot_speak(history):
        if history and history[-1][1]: return speak(history[-1][1])
        return None

    def reset_media_slots():
        return None, None, None, ""

    def do_clear(sid):
        clear_history(sid)
        return [], [], "", None, None, None

    msg.submit(
        user_send, [msg, file_context_state, camera_input, canvas_input, chat_state, session_id], [msg, chatbot, chat_state]
    ).then(
        bot_reply, [chat_state, system_prompt, temperature, max_tokens, file_context_state, research_toggle, session_id], [chatbot, chat_state]
    ).then(
        bot_speak, [chat_state], [reply_audio]
    ).then(
        reset_media_slots, None, [file_input, camera_input, canvas_input, file_context_state]
    )

    send_btn.click(
        user_send, [msg, file_context_state, camera_input, canvas_input, chat_state, session_id], [msg, chatbot, chat_state]
    ).then(
        bot_reply, [chat_state, system_prompt, temperature, max_tokens, file_context_state, research_toggle, session_id], [chatbot, chat_state]
    ).then(
        bot_speak, [chat_state], [reply_audio]
    ).then(
        reset_media_slots, None, [file_input, camera_input, canvas_input, file_context_state]
    )

    clear_btn.click(do_clear, [session_id], [chatbot, chat_state, file_context_state, file_input, camera_input, canvas_input])

port_number = int(os.environ.get("PORT", 10000))
demo.queue(default_concurrency_limit=4).launch(server_name="0.0.0.0", server_port=port_number)
