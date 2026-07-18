import os
import tempfile
import sqlite3
import gradio as gr
from huggingface_hub import InferenceClient

# Securely retrieve your Hugging Face token
hf_token = os.environ.get("HF_TOKEN")

# Initialize the Hugging Face clients
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=hf_token)
whisper_client = InferenceClient("openai/whisper-large-v3", token=hf_token)
tts_client = InferenceClient("microsoft/speecht5_tts", token=hf_token)

# --- SESSION MEMORY (SQLite) ---
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
        for turn in history:
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

# --- UI DESIGN (v11.0 INTERACTIVE ROW PLATFORM) ---

custom_css = """
footer {visibility: hidden}
.gradio-container {background-color: #0b0f19;}
.inline-row > div {align-self: center !important;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"), css=custom_css) as demo:
    gr.Markdown("# ⚡ THUNDER WORKSPACE // Core v11.0")

    chatbot = gr.Chatbot(value=load_history(), height=460)

    # UNIFIED SINGLE MULTI-FEATURE INTERACTIVE ROW
    with gr.Row(elem_classes=["inline-row"]):
        msg = gr.Textbox(placeholder="Type message...", show_label=False, scale=5, container=False)
        send_btn = gr.Button("Send", scale=1, min_width=60)
        audio_input = gr.Audio(sources=["microphone"], type="filepath", show_label=False, scale=2, container=False)
        file_input = gr.File(show_label=False, scale=2, file_count="single", container=False)
        search_toggle = gr.Checkbox(label="🔍 Search", value=False, scale=1)
        
        # New Feature Node: Collapsible settings inside the single line
        settings_toggle = gr.Checkbox(label="⚙️ Settings", value=False, scale=1)
        clear_btn = gr.Button("🗑️ Clear", variant="stop", scale=1, min_width=60)

    # Dynamic settings workspace container controlled directly by the row button
    with gr.Box(visible=False) as settings_panel:
        with gr.Row():
            system_prompt = gr.Textbox(value=DEFAULT_SYSTEM_PROMPT, label="System Directives / Brain Prompt", lines=2, scale=6)
            temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.75, step=0.05, label="Temperature Node", scale=3)
            max_tokens = gr.Slider(minimum=256, maximum=4096, value=1024, step=128, label="Max Generated Tokens", scale=3)

    reply_audio = gr.Audio(label="🔊 Vocal Feedback Node", autoplay=True, visible=False)
    file_context = gr.State("")

    # Bind toggle visibility to configuration layout
    def toggle_panel(visible):
        return gr.update(visible=visible)
    
    settings_toggle.change(toggle_panel, inputs=[settings_toggle], outputs=[settings_panel])

    # Actions binding
    file_input.change(read_file, inputs=[file_input], outputs=[file_context])
    audio_input.change(transcribe, inputs=[audio_input], outputs=[msg])

    def user_send(message, history):
        message = (message or "").strip()
        if not message:
            return "", history
        save_message("user", message)
        return "", history + [[message, ""]]

    def bot_reply(history, sys_prompt, temp, tokens, f_context, search_on):
        if not history:
            yield history
            return
        message = history[-1][0]
        api_history = history[:-1]
        
        for chunk in respond(message, api_history, sys_prompt, temp, tokens, f_context, search_on):
            history[-1][1] = chunk
            yield history
        save_message("assistant", history[-1][1])

    def bot_speak(history):
        if history and history[-1][1]:
            return speak(history[-1][1])
        return None

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
demo.queue(default_concurrency_limit=3).launch(server_name="0.0.0.0", server_port=port_number)
