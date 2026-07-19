import os
import uuid
import tempfile
import sqlite3
import threading
import gradio as gr

from huggingface_hub import InferenceClient
from google import genai
from google.genai import types
from anthropic import Anthropic

# Securely retrieve your API tokens
hf_token = os.environ.get("HF_TOKEN")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")

# Initialize the Hugging Face clients
whisper_client = InferenceClient("openai/whisper-large-v3", token=hf_token)
tts_client = InferenceClient("microsoft/speecht5_tts", token=hf_token)

# Initialize Gemini + Claude clients
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
claude_client = Anthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY else None

# How many prior turns to send to the model
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
            text = "
".join(page.extract_text() or "" for page in reader.pages)
        elif ext in (".txt", ".md", ".csv", ".py", ".json"):
            with open(file_name, "r", errors="ignore") as f:
                text = f.read()
        else:
            return f"[Unsupported file type: {ext}]"

        max_chars = 12000
        if len(text) > max_chars:
            text = text[:max_chars] + "
...[truncated]"
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
        return "

".join(
            f"- {r.get('title','')}: {r.get('body','')} ({r.get('href','')})"
            for r in results
        )
    except Exception as e:
        return f"[Search Error: {str(e)}]"

def trim_history(history):
    return history[-MAX_CONTEXT_TURNS:] if len(history) > MAX_CONTEXT_TURNS else history

def build_context(system_prompt, file_context, search_context):
    full = system_prompt
    if search_context:
        full += "

Live web search results:

" + search_context
    if file_context:
        full += "

Uploaded workspace file context:

" + file_context
    return full

def choose_model(message, history):
    text = (message or "").lower().strip()
    words = len(text.split())

    if any(k in text for k in ["write", "rewrite", "polish", "improve", "edit", "essay", "email", "report"]):
        return "Claude"
    if any(k in text for k in ["what is", "why", "how does", "explain", "define", "summarize"]):
        return "Gemini"
    if words <= 12:
        return "Gemini"
    if len(history) >= 6:
        return "Claude"
    if len(text) > 180:
        return "Claude"
    return "Gemini"

def gemini_answer(message, history, system_prompt, temperature, max_tokens):
    if not gemini_client:
        return "[Gemini API key missing]"
    try:
        contents = []
        for turn in history:
            role = "user" if turn["role"] == "user" else "model"
            contents.append({"role": role, "parts": [turn["content"]]})
        contents.append({"role": "user", "parts": [message]})

        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=float(temperature),
                max_output_tokens=int(max_tokens),
            ),
        )
        return response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        return f"[Gemini Error: {str(e)}]"

def claude_answer(message, history, system_prompt, temperature, max_tokens):
    if not claude_client:
        return "[Claude API key missing]"
    try:
        messages = []
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        response = claude_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            system=system_prompt,
            messages=messages + [{"role": "user", "content": message}],
        )
        return response.content[0].text
    except Exception as e:
        return f"[Claude Error: {str(e)}]"

def respond(message, history, system_prompt, temperature, max_tokens, file_context, search_enabled):
    try:
        search_context = ""
        if search_enabled:
            results = web_search(message)
            if results and not results.startswith("[Search Error"):
                search_context = results

        full_system_prompt = build_context(system_prompt, file_context, search_context)
        trimmed_history = trim_history(history)
        model_name = choose_model(message, trimmed_history)

        if model_name == "Gemini":
            answer = gemini_answer(message, trimmed_history, full_system_prompt, temperature, max_tokens)
        else:
            answer = claude_answer(message, trimmed_history, full_system_prompt, temperature, max_tokens)

        return answer, model_name
    except Exception as e:
        return f"Error: {str(e)}", "Error"

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

    session_id = gr.State(None)

    chatbot = gr.Chatbot(
        value=[],
        height=440,
        elem_id="chat-panel",
        placeholder="**Thunder is ready.** Ask a question, drop a file, or hit the mic to get started.",
        avatar_images=(None, None),
    )

    with gr.Row():
        msg = gr.Textbox(placeholder="Type message or stream voice data input...", scale=8)
        send_btn = gr.Button("Send", scale=2)

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Mic Input", scale=3)
        file_input = gr.File(label="📎 Attachment (.pdf, .txt, .csv, .md, .json)", scale=4)

        with gr.Column(scale=3, min_width=150):
            search_toggle = gr.Checkbox(label="🔍 Search web on prompt", value=False)
            clear_btn = gr.Button("🆕 New chat", variant="stop")

    with gr.Accordion("⚙️ Core Configurations & Engine Settings", open=False):
        with gr.Row():
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Directives / Brain Prompt", lines=2, scale=6)
            temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.75, step=0.05, label="Temperature Node", scale=3)
            max_tokens = gr.Slider(minimum=256, maximum=4096, value=1024, step=128, label="Max Generated Tokens", scale=3)

    model_label = gr.Markdown("")
    reply_audio = gr.Audio(label="🔊 Vocal Feedback Node", autoplay=True)
    file_context = gr.State("")

    def start_session():
        new_id = str(uuid.uuid4())
        return new_id, load_history(new_id), load_history(new_id), ""

    demo.load(start_session, None, [session_id, chatbot, chatbot, model_label])

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
            yield history, "Selected model: none"
            return

        message = history[-1][0]
        api_history = [{"role": "user", "content": t[0]} if i % 2 == 0 else {"role": "assistant", "content": t[1]}
                       for i, t in enumerate([])]
        api_history = history[:-1]
        answer, model_name = respond(message, api_history, sys_prompt, temp, tokens, f_context, search_on)

        history[-1][1] = answer
        yield history, f"Selected model: **{model_name}**"
        save_message(sid, "assistant", answer)

    def bot_speak(history):
        if history and history[-1][1]:
            return speak(history[-1][1])
        return None

    def do_clear(sid):
        clear_history(sid)
        return [], ""

    msg.submit(user_send, [msg, chatbot, session_id], [msg, chatbot]).then(
        bot_reply, [chatbot, system_prompt, temperature, max_tokens, file_context, search_toggle, session_id], [chatbot, model_label]
    ).then(
        bot_speak, chatbot, reply_audio
    )

    send_btn.click(user_send, [msg, chatbot, session_id], [msg, chatbot]).then(
        bot_reply, [chatbot, system_prompt, temperature, max_tokens, file_context, search_toggle, session_id], [chatbot, model_label]
    ).then(
        bot_speak, chatbot, reply_audio
    )

    clear_btn.click(do_clear, [session_id], [chatbot, model_label])

port_number = int(os.environ.get("PORT", 10000))
demo.queue(default_concurrency_limit=3).launch(server_name="0.0.0.0", server_port=port_number)
