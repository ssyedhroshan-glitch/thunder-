import os
import tempfile
import sqlite3
import json
import gradio as gr
from huggingface_hub import InferenceClient

# Securely retrieve your Hugging Face token
hf_token = os.environ.get("HF_TOKEN")

# Initialize the Hugging Face clients
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=hf_token)
whisper_client = InferenceClient("openai/whisper-large-v3", token=hf_token)
tts_client = InferenceClient("microsoft/speecht5_tts", token=hf_token)

# --- SESSION MEMORY (SQLite) ---
# NOTE: Render's free tier wipes disk on every restart/spin-down, so this
# memory only survives while the app stays awake. For memory that survives
# redeploys, swap this out for an external DB (e.g. Supabase/Postgres) later.
DB_PATH = "thunder_memory.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

def load_history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT role, content FROM history ORDER BY id").fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in rows]

def save_message(role, content):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO history (role, content) VALUES (?, ?)", (role, content))
    conn.commit()
    conn.close()

def clear_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM history")
    conn.commit()
    conn.close()

init_db()

# The Brain: Custom peer system prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder, an elite, tech-savvy AI collaborator with a sharp mind and a touch of dry wit. "
    "You talk to the user as a brilliant, supportive peer and co-founder. "
    "Provide highly insightful, direct, and scannable answers using bold headers and clean bullet points. "
    "Break down complex concepts with high technical accuracy but zero dry academic jargon. "
    "Keep your tone authentic, grounded, and engaging. Never use robotic disclaimers. "
    "Only bring up a specific topic (aerospace, finance, code, etc.) if the user actually raises it."
)

# --- HELPER FUNCTIONS ---

def transcribe(audio_path):
    """Voice input: turn recorded speech into text."""
    if audio_path is None:
        return ""
    try:
        result = whisper_client.automatic_speech_recognition(audio_path)
        return result.text
    except Exception as e:
        return f"[Transcription Error: {str(e)}]"


def speak(text):
    """Voice output: turn Thunder's reply into playable audio."""
    if not text:
        return None
    try:
        audio_bytes = tts_client.text_to_speech(text)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name
    except Exception:
        return None


def read_file(file_path):
    """File reading: extract text from an uploaded file so Thunder can use it as context."""
    if file_path is None:
        return ""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext in (".txt", ".md", ".csv", ".py", ".json"):
            with open(file_path, "r", errors="ignore") as f:
                text = f.read()
        else:
            return f"[Unsupported file type: {ext}]"

        max_chars = 12000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...[truncated]"
        return text
    except Exception as e:
        return f"[File Read Error: {str(e)}]"


def web_search(query, max_results=5):
    """Live web search using DuckDuckGo (no API key needed)."""
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return ""
        return "\n\n".join(
            f"- {r.get('title','')}: {r.get('body','')} ({r.get('href','')})"
            for r in results
        )
    except Exception as e:
        return f"[Search Error: {str(e)}]"


def respond(message, history, system_prompt, temperature, max_tokens, file_context, search_enabled):
    try:
        full_system_prompt = system_prompt

        if search_enabled:
            results = web_search(message)
            if results and not results.startswith("[Search Error"):
                full_system_prompt += (
                    "\n\nHere are live web search results relevant to the user's message. "
                    "Use them if helpful, and mention they're current info when relevant:\n\n" + results
                )

        if file_context:
            full_system_prompt += (
                "\n\nThe user has shared a file. Use its contents to answer questions "
                "when relevant:\n\n" + file_context
            )

        messages = [{"role": "system", "content": full_system_prompt}]

        for item in history:
            role = item.get("role")
            content = item.get("content")
            if role and content:
                messages.append({"role": role, "content": content})

        messages.append({"role": "user", "content": message})

        response = ""
        for token in client.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        ):
            token_text = token.choices[0].delta.content
            if token_text:
                response += token_text
                yield response

    except Exception as e:
        yield f"Error: {str(e)}"


# --- CUSTOM UI WITH GR.BLOCKS ---

custom_css = """
footer {visibility: hidden}
.gradio-container {background-color: #0b0f19;}
"""

with gr.Blocks() as demo:
    gr.Markdown("# ⚡ THUNDER WORKSPACE // Voice + Files + Search v7")
    gr.Markdown("Speak, type, upload a file, or search the web — Thunder handles it all.")

    chatbot = gr.Chatbot(value=load_history())

    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message here or speak into the microphone...", scale=7)
        send_btn = gr.Button("Send", scale=1)
        audio_input = gr.Audio(sources=["microphone"], type="filepath", scale=4)

    with gr.Row():
        file_input = gr.File(label="📎 Upload a file (.pdf, .txt, .csv, .md, .json)", scale=6)
        search_toggle = gr.Checkbox(label="🔍 Search the web for this", scale=2)
        clear_btn = gr.Button("🗑️ Clear memory", scale=2)

    reply_audio = gr.Audio(label="🔊 Thunder's voice", autoplay=True)

    audio_input.change(transcribe, inputs=[audio_input], outputs=[msg])

    # State
    system_prompt = gr.State(DEFAULT_SYSTEM_PROMPT)
    temperature = gr.State(0.75)
    max_tokens = gr.State(1024)
    file_context = gr.State("")

    file_input.change(read_file, inputs=[file_input], outputs=[file_context])

    def user_send(message, history):
        history = history + [{"role": "user", "content": message}]
        save_message("user", message)
        return "", history

    def bot_reply(history, sys_prompt, temp, tokens, f_context, search_on):
        message = history[-1]["content"]
        history.append({"role": "assistant", "content": ""})
        for chunk in respond(message, history[:-1], sys_prompt, temp, tokens, f_context, search_on):
            history[-1]["content"] = chunk
            yield history
        save_message("assistant", history[-1]["content"])

    def bot_speak(history):
        last_reply = history[-1]["content"] if history else ""
        return speak(last_reply)

    def do_clear():
        clear_history()
        return []

    msg.submit(user_send, [msg, chatbot], [msg, chatbot]).then(
        bot_reply, [chatbot, system_prompt, temperature, max_tokens, file_context, search_toggle], chatbot
    ).then(
        bot_speak, chatbot, reply_audio
    )

    send_btn.click(user_send, [msg, chatbot], [msg, chatbot]).then(
        bot_reply, [chatbot, system_prompt, temperature, max_tokens, file_context, search_toggle], chatbot
    ).then(
        bot_speak, chatbot, reply_audio
    )

    clear_btn.click(do_clear, None, chatbot)

port_number = int(os.environ.get("PORT", 10000))
demo.launch(
    server_name="0.0.0.0",
    server_port=port_number,
    theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"),
    css=custom_css,
    )
