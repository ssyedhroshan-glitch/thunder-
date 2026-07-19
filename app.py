import os
import uuid
import tempfile
import sqlite3
import gradio as gr
from huggingface_hub import InferenceClient

hf_token = os.environ.get("HF_TOKEN")

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
    conn.execute(
        "INSERT INTO history (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

def clear_history(session_id):
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
    "Break down complex concepts clearly. "
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
        clean_text = text.replace("*", "").replace("#", "").replace("`", "")
        clean_text = clean_text[:200]
        audio_bytes = tts_client.text_to_speech(clean_text)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name
    except Exception:
        return None

def web_search(query, max_results=3):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        lines = []
        for r in results[:max_results]:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            lines.append(f"- {title}: {body} ({href})")
        return "\n".join(lines)
    except Exception as e:
        return f"[Search Error: {e}]"

def build_messages(chat_history, system_prompt, search_context):
    messages = [{"role": "system", "content": system_prompt}]
    if search_context:
        messages.append({"role": "system", "content": "Live Web Search Results:\n\n" + search_context})
    
    for u, a in chat_history:
        if u:
            messages.append({"role": "user", "content": u})
        if a:
            messages.append({"role": "assistant", "content": a})
    return messages

def stream_reply(messages, temperature, max_tokens):
    text = ""
    for chunk in client.chat.completions.create(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    ):
        part = chunk.choices[0].delta.content or ""
        text += part
        yield text

custom_css = """
footer {visibility: hidden;}
body, .gradio-container {background-color: #0b0f19 !important;}
.panel-card {
    background-color: #121826 !important; 
    border: 1px solid #1f293d !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
.chatbot-container {
    border: 1px solid #1f293d !important;
    border-radius: 12px !important;
}
/* Premium layout alignment targeting for custom control row */
.top-bar-right {
    display: flex !important;
    justify-content: flex-end !important;
    align-items: center !important;
    gap: 15px !important;
}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"), 
    css=custom_css,
    js="() => { document.querySelector('body').classList.add('dark'); }"
) as demo:
    
    session_id = gr.State(None)
    chat_state = gr.State([])
    
    # HEADER & TOP RIGHT SETTINGS CONTROL LAYER
    with gr.Row():
        with gr.Column(scale=6):
            gr.Markdown("# ⚡ THUNDER WORKSPACE")
        with gr.Column(scale=4, elem_classes=["top-bar-right"]):
            search_toggle = gr.Checkbox(label="🔍 Search Mode", value=False, container=False)
            settings_toggle = gr.Checkbox(label="⚙️ Settings", value=False, container=False)
            clear_btn = gr.Button("🆕 Reset", variant="stop", size="sm")

    # EXPANDABLE TOP RUNTIME PARAMETERS DRAWER
    with gr.Group(visible=False, elem_classes=["panel-card"]) as settings_panel:
        with gr.Row():
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Directives", lines=2)
            temperature = gr.Slider(0.1, 1.5, value=0.70, step=0.05, label="Temperature")
            max_tokens = gr.Slider(256, 4096, value=1536, step=128, label="Token capacity")

    chatbot = gr.Chatbot(height=500, elem_classes=["chatbot-container"])

    # HIGHSPEED CONSOLE CONSTRUCT WITH MOCK FILE DROPPED PORT FOR AUDIO INGESTION
    with gr.Row():
        with gr.Column(scale=1, min_width=60):
            # Custom styled hidden trigger component acting as the microphone portal button
            audio_input = gr.Audio(sources=["microphone"], type="filepath", label="🎤", container=False)
        with gr.Column(scale=10):
            msg = gr.Textbox(
                show_label=False,
                placeholder="Send a command or tap microphone to speak...",
                container=False
            )
        with gr.Column(scale=1, min_width=60):
            send_btn = gr.Button("⚡", variant="primary")

    reply_audio = gr.Audio(autoplay=True, visible=False)

    def start_session():
        sid = str(uuid.uuid4())
        initial_history = load_history(sid)
        return sid, initial_history, initial_history

    demo.load(start_session, None, [session_id, chatbot, chat_state])

    settings_toggle.change(lambda visible: gr.update(visible=visible), inputs=[settings_toggle], outputs=[settings_panel])

    def on_audio(audio_path):
        if not audio_path:
            return gr.update()
        text = transcribe(audio_path)
        return text

    def user_send(message, history, sid):
        message = (message or "").strip()
        if not message:
            return "", history, history
        history = history + [[message, ""]]
        save_message(sid, "user", message)
        return "", history, history

    def bot_reply(history, sys_prompt, temp, tokens, search_on, sid):
        if not history:
            return history, history

        last_user = history[-1][0]
        search_context = web_search(last_user) if search_on else ""
        
        messages = build_messages(history[:-1], sys_prompt, search_context)
        messages.append({"role": "user", "content": last_user})

        final_text = ""
        try:
            for partial in stream_reply(messages, temp, tokens):
                final_text = partial
                history[-1][1] = final_text
                yield history, history
            save_message(sid, "assistant", final_text)
        except Exception as e:
            err = f"Error: {e}"
            history[-1][1] = err
            save_message(sid, "assistant", err)
            yield history, history

    def bot_speak(history):
        if history and history[-1][1]:
            return speak(history[-1][1])
        return None

    def do_clear(sid):
        clear_history(sid)
        return [], []

    audio_input.change(on_audio, inputs=[audio_input], outputs=[msg])

    msg.submit(
        user_send, [msg, chat_state, session_id], [msg, chatbot, chat_state]
    ).then(
        bot_reply, [chat_state, system_prompt, temperature, max_tokens, search_toggle, session_id], [chatbot, chat_state]
    ).then(
        bot_speak, [chat_state], [reply_audio]
    )

    send_btn.click(
        user_send, [msg, chat_state, session_id], [msg, chatbot, chat_state]
    ).then(
        bot_reply, [chat_state, system_prompt, temperature, max_tokens, search_toggle, session_id], [chatbot, chat_state]
    ).then(
        bot_speak, [chat_state], [reply_audio]
    )

    clear_btn.click(do_clear, [session_id], [chatbot, chat_state])

port_number = int(os.environ.get("PORT", 10000))
demo.queue(default_concurrency_limit=4).launch(server_name="0.0.0.0", server_port=port_number)
    
