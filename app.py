import os
import uuid
import tempfile
import sqlite3
import gradio as gr

from google import genai
from anthropic import Anthropic

# -------------------------
# ENV KEYS
# -------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY")

# -------------------------
# CLIENTS
# -------------------------
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None
claude_client = Anthropic(api_key=CLAUDE_API_KEY) if CLAUDE_API_KEY else None

# -------------------------
# SQLITE SESSION MEMORY
# -------------------------
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
    return [{"role": role, "content": content} for role, content in rows]

init_db()

# -------------------------
# SYSTEM PROMPT
# -------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder, an elite, tech-savvy AI collaborator with a sharp mind and a touch of dry wit. "
    "You talk to the user as a brilliant, supportive peer and co-founder. "
    "Provide highly insightful, direct, and scannable answers using short headings and clean bullet points. "
    "Break down complex concepts clearly. "
    "Keep your tone authentic, grounded, and engaging."
)

# -------------------------
# HELPERS
# -------------------------
def transcribe(audio_path):
    if not audio_path:
        return ""
    try:
        from huggingface_hub import InferenceClient
        whisper_client = InferenceClient(model="openai/whisper-large-v3", token=os.environ.get("HF_TOKEN"))
        result = whisper_client.automatic_speech_recognition(audio_path)
        return result.text if hasattr(result, "text") else str(result)
    except Exception as e:
        return f"[Transcription Error: {e}]"

def speak(text):
    if not text:
        return None
    try:
        from huggingface_hub import InferenceClient
        tts_client = InferenceClient(model="microsoft/speecht5_tts", token=os.environ.get("HF_TOKEN"))
        clean_text = text.replace("*", "").replace("#", "").replace("`", "")
        clean_text = clean_text[:250]
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
            text = "
".join(page.extract_text() or "" for page in reader.pages)
        elif ext in (".txt", ".md", ".csv", ".py", ".json"):
            with open(file_path, "r", errors="ignore") as f:
                text = f.read()
        else:
            return f"[Unsupported file type: {ext}]"

        return text[:12000]
    except Exception as e:
        return f"[File Read Error: {e}]"

def build_context(system_prompt, file_context):
    prompt = system_prompt
    if file_context:
        prompt += "

Uploaded file context:

" + file_context
    return prompt

def gemini_answer(message, history, system_prompt, temperature, max_tokens, file_context):
    if not gemini_client:
        return "[Gemini API key missing]"
    try:
        contents = []
        for turn in history:
            contents.append({"role": turn["role"], "parts": [turn["content"]]})
        contents.append({"role": "user", "parts": [message]})

        config = {
            "temperature": float(temperature),
            "max_output_tokens": int(max_tokens),
            "system_instruction": build_context(system_prompt, file_context),
        }

        response = gemini_client.models.generate_content(
            model="gemini-1.5-flash",
            contents=contents,
            config=config,
        )
        return response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        return f"[Gemini Error: {e}]"

def claude_answer(message, history, system_prompt, temperature, max_tokens, file_context):
    if not claude_client:
        return "[Claude API key missing]"
    try:
        messages = []
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["content"]})

        system_text = build_context(system_prompt, file_context)

        response = claude_client.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            system=system_text,
            messages=messages + [{"role": "user", "content": message}],
        )
        return response.content[0].text
    except Exception as e:
        return f"[Claude Error: {e}]"

def respond(message, history, system_prompt, temperature, max_tokens, file_context, model_choice):
    if model_choice == "Gemini":
        return gemini_answer(message, history, system_prompt, temperature, max_tokens, file_context)
    return claude_answer(message, history, system_prompt, temperature, max_tokens, file_context)

# -------------------------
# GRADIO UI
# -------------------------
custom_css = """
footer {visibility: hidden;}
.gradio-container {background-color: #0b0f19;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"), css=custom_css) as demo:
    gr.Markdown("# ⚡ THUNDER WORKSPACE // Gemini + Claude Router")
    gr.Markdown("Choose a model manually, then chat with voice, files, and session memory.")

    session_id = gr.State(None)
    chat_state = gr.State([])
    file_context = gr.State("")

    chatbot = gr.Chatbot(type="messages", height=440)

    with gr.Row():
        msg = gr.Textbox(placeholder="Type message or use mic...", scale=8)
        send_btn = gr.Button("Send", scale=2)

    with gr.Row():
        model_choice = gr.Radio(
            choices=["Gemini", "Claude"],
            value="Gemini",
            label="Choose model"
        )
        search_toggle = gr.Checkbox(label="🔍 Search web on prompt", value=False)

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Mic Input")
        file_input = gr.File(label="📎 Attachment (.pdf, .txt, .csv, .md, .json)")
        clear_btn = gr.Button("🆕 New chat", variant="stop")

    with gr.Accordion("⚙️ Settings", open=False):
        system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System prompt", lines=3)
        temperature = gr.Slider(0.1, 1.5, value=0.75, step=0.05, label="Temperature")
        max_tokens = gr.Slider(256, 4096, value=1024, step=128, label="Max tokens")

    reply_audio = gr.Audio(label="🔊 Vocal Feedback", autoplay=True)

    def start_session():
        sid = str(uuid.uuid4())
        return sid, load_history(sid), load_history(sid)

    demo.load(start_session, None, [session_id, chatbot, chat_state])

    def on_audio(audio_path):
        return transcribe(audio_path)

    def on_file(file_obj):
        return read_file(file_obj)

    def user_send(message, history, sid):
        message = (message or "").strip()
        if not message:
            return "", history, history
        history = history + [{"role": "user", "content": message}]
        save_message(sid, "user", message)
        return "", history, history

    def bot_reply(history, sys_prompt, temp, tokens, f_context, model_name, sid):
        if not history:
            return history, history

        user_message = history[-1]["content"]
        answer = respond(
            user_message,
            history[:-1],
            sys_prompt,
            temp,
            tokens,
            f_context,
            model_name
        )

        history = history + [{"role": "assistant", "content": answer}]
        save_message(sid, "assistant", answer)
        return history, history

    def bot_speak(history):
        if history and history[-1].get("content"):
            return speak(history[-1]["content"])
        return None

    def do_clear(sid):
        clear_history(sid)
        return [], []

    audio_input.change(on_audio, inputs=[audio_input], outputs=[msg])
    file_input.change(on_file, inputs=[file_input], outputs=[file_context])

    msg.submit(
        user_send, [msg, chat_state, session_id], [msg, chatbot, chat_state]
    ).then(
        bot_reply, [chat_state, system_prompt, temperature, max_tokens, file_context, model_choice, session_id], [chatbot, chat_state]
    ).then(
        bot_speak, [chat_state], [reply_audio]
    )

    send_btn.click(
        user_send, [msg, chat_state, session_id], [msg, chatbot, chat_state]
    ).then(
        bot_reply, [chat_state, system_prompt, temperature, max_tokens, file_context, model_choice, session_id], [chatbot, chat_state]
    ).then(
        bot_speak, [chat_state], [reply_audio]
    )

    clear_btn.click(do_clear, [session_id], [chatbot, chat_state])

port_number = int(os.environ.get("PORT", 10000))
demo.queue(default_concurrency_limit=3).launch(server_name="0.0.0.0", server_port=port_number)
