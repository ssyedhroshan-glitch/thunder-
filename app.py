import os
import uuid
import tempfile
import sqlite3
import threading
import gradio as gr
from huggingface_hub import InferenceClient

# Securely retrieve your Hugging Face token
hf_token = os.environ.get("HF_TOKEN")

# Initialize the Hugging Face clients
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=hf_token)
whisper_client = InferenceClient("openai/whisper-large-v3", token=hf_token)
tts_client = InferenceClient("microsoft/speecht5_tts", token=hf_token)

# How many prior turns to send to the model (keeps latency/cost bounded on long chats)
MAX_CONTEXT_TURNS = 12

# --- SESSION MEMORY (SQLite, single reusable connection, WAL mode for concurrency) ---
DB_PATH = "thunder_memory.db"
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
                role TEXT,
                content TEXT
            )
        """)
        _conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON history(session_id)")
        _conn.commit()

def load_history(session_id):
    with _db_lock:
        rows = _conn.execute(
            "SELECT role, content FROM history WHERE session_id = ? ORDER BY id",
            (session_id,)
        ).fetchall()

    chat_list = []
    current_pair = [None, None]
    for role, content in rows:
        if role == "user":
            if current_pair[0] is not None:
                chat_list.append(current_pair)
                current_pair = [None, None]
            current_pair[0] = content
        elif role == "assistant":
            current_pair[1] = content
            chat_list.append(current_pair)
            current_pair = [None, None]
    if current_pair[0] is not None:
        chat_list.append(current_pair)
    return chat_list

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
    "You are Thunder, an elite, tech-savvy AI collaborator with a sharp mind and a touch of dry wit. "
    "You talk to the user as a brilliant, supportive peer and co-founder. "
    "Provide highly insightful, direct, and scannable answers using bold headers and clean bullet points. "
    "Break down complex concepts with high technical accuracy but zero dry academic jargon. "
    "Keep your tone authentic, grounded, and engaging. Never use robotic disclaimers. "
    "Only bring up a specific topic if the user raises it."
)

# --- HELPER FUNCTIONS ---

def transcribe(audio_path):
    if audio_path is None:
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
        clean_text = text.replace("*", "").replace("#", "").replace("- ", "").replace("`", "")
        if len(clean_text) > 250:
            clean_text = clean_text[:250] + "..."
        audio_bytes = tts_client.text_to_speech(clean_text)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name
    except Exception:
        return None

def read_file(file_path):
    if file_path is None:
        return ""
    try:
        file_name = file_path.name if hasattr(file_path, "name") else file_path
        ext = os.path.splitext(file_name)[1].lower()
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_name)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext in (".txt", ".md", ".csv", ".py", ".json"):
            with open(file_name, "r", errors="ignore") as f:
                text = f.read()
        else:
            return f"[Unsupported file type: {ext}]"

        max_chars = 12000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text
    except Exception as e:
        return f"[File Read Error: {str(e)}]"

def web_search(query, max_results=3):
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        return "\n\n".join(f"- {r.get('title','')}: {r.get('body','')} ({r.get('href','')})" for r in results)
    except Exception as e:
        return f"[Search Error: {str(e)}]"

def respond(message, history, system_prompt, temperature, max_tokens, file_context, search_enabled):
    try:
        full_system_prompt = system_prompt
        if search_enabled:
            results = web_search(message)
            if results and not results.startswith("[Search Error"):
                full_system_prompt += "\n\nLive web search results:\n\n" + results

        if file_context:
            full_system_prompt += "\n\nUploaded workspace file context:\n\n" + file_context

        messages = [{"role": "system", "content": full_system_prompt}]
        trimmed_history = history[-MAX_CONTEXT_TURNS:] if len(history) > MAX_CONTEXT_TURNS else history
        for turn in trimmed_history:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                if turn[0]: messages.append({"role": "user", "content": turn[0]})
                if turn[1]: messages.append({"role": "assistant", "content": turn[1]})

        messages.append({"role": "user", "content": message})

        response = ""
        for token in client.chat_completion(messages, max_tokens=max_tokens, temperature=temperature, stream=True):
            token_text = token.choices[0].delta.content
            if token_text:
                response += token_text
                yield response
    except Exception as e:
        yield f"Error: {str(e)}"

# --- UI DESIGN (v9.5 SESSION-ISOLATED WORKSPACE) ---

custom_css = """
footer {visibility: hidden}
.gradio-container {
    background: radial-gradient(circle at 20% -10%, #132030 0%, #0a0e17 45%, #05070c 100%);
}
#thunder-header h1 {
    font-weight: 700;
    background: linear-gradient(90deg, #22d3ee, #818cf8);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    letter-spacing: 0.5px;
}
#thunder-header p {
    color: #8b95a7;
}
.gr-box, .block {
    border-radius: 14px !important;
}
#chat-panel {
    border: 1px solid rgba(34, 211, 238, 0.15);
    border-radius: 16px;
    box-shadow: 0 0 24px rgba(34, 211, 238, 0.05);
}
button.primary {
    background: linear-gradient(90deg, #06b6d4, #6366f1) !important;
    border: none !important;
}
button.stop {
    background: rgba(244, 63, 94, 0.12) !important;
    color: #fb7185 !important;
    border: 1px solid rgba(244, 63, 94, 0.3) !important;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"), css=custom_css) as demo:
    with gr.Column(elem_id="thunder-header"):
        gr.Markdown("# ⚡ Thunder Workspace")
        gr.Markdown("Your private AI collaborator — voice, files, and live search in one place. Each visitor gets an isolated, persistent session.")

    # One unique session_id per browser tab, generated fresh on page load
    session_id = gr.State(None)

    chatbot = gr.Chatbot(
        value=[],
        height=440,
        elem_id="chat-panel",
        placeholder="**Thunder is ready.** Ask a question, drop a file, or hit the mic to get started.",
        avatar_images=(None, None),
    )

    # Line 1: Main Text input & Send Array
    with gr.Row():
        msg = gr.Textbox(placeholder="Type message or stream voice data input...", scale=8)
        send_btn = gr.Button("Send", scale=2)

    # Line 2: Balanced Core Features Row
    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Mic Input", scale=3)
        file_input = gr.File(label="📎 Attachment (.pdf, .txt, .csv, .md, .json)", scale=4)

        with gr.Column(scale=3, min_width=150):
            search_toggle = gr.Checkbox(label="🔍 Search web on prompt", value=False)
            clear_btn = gr.Button("🆕 New chat", variant="stop")

    # Expandable Settings Drawer Configuration Block
    with gr.Accordion("⚙️ Core Configurations & Engine Settings", open=False):
        with gr.Row():
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Directives / Brain Prompt", lines=2, scale=6)
            temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.75, step=0.05, label="Temperature Node", scale=3)
            max_tokens = gr.Slider(minimum=256, maximum=4096, value=1024, step=128, label="Max Generated Tokens", scale=3)

    reply_audio = gr.Audio(label="🔊 Vocal Feedback Node", autoplay=True)
    file_context = gr.State("")

    # --- Session bootstrap: give each tab its own UUID and load ONLY its own history ---
    def start_session():
        new_id = str(uuid.uuid4())
        return new_id, load_history(new_id)

    demo.load(start_session, None, [session_id, chatbot])

    # Actions binding
    file_input.change(read_file, inputs=[file_input], outputs=[file_context])
    audio_input.change(transcribe, inputs=[audio_input], outputs=[msg])

    def user_send(message, history, sid):
        message = (message or "").strip()
        if not message:
            return "", history
        save_message(sid, "user", message)
        return "", history + [[message, ""]]

    def bot_reply(history, sys_prompt, temp, tokens, f_context, search_on, sid):
        if not history:
            yield history
            return
        message = history[-1][0]
        api_history = history[:-1]

        for chunk in respond(message, api_history, sys_prompt, temp, tokens, f_context, search_on):
            history[-1][1] = chunk
            yield history
        save_message(sid, "assistant", history[-1][1])

    def bot_speak(history):
        if history and history[-1][1]:
            return speak(history[-1][1])
        return None

    def do_clear(sid):
        clear_history(sid)
        return []

    msg.submit(user_send, [msg, chatbot, session_id], [msg, chatbot]).then(
        bot_reply, [chatbot, system_prompt, temperature, max_tokens, file_context, search_toggle, session_id], chatbot
    ).then(
        bot_speak, chatbot, reply_audio
    )

    send_btn.click(user_send, [msg, chatbot, session_id], [msg, chatbot]).then(
        bot_reply, [chatbot, system_prompt, temperature, max_tokens, file_context, search_toggle, session_id], chatbot
    ).then(
        bot_speak, chatbot, reply_audio
    )

    clear_btn.click(do_clear, [session_id], chatbot)

port_number = int(os.environ.get("PORT", 10000))
demo.queue(default_concurrency_limit=3).launch(server_name="0.0.0.0", server_port=port_number)
