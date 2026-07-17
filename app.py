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

def load_history_as_list():
    """Loads history in the exact format Gradio natively expects: [[user, bot]]."""
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

# The Brain: Custom peer system prompt
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
    """Voice input: turn recorded speech into text."""
    if audio_path is None:
        return ""
    try:
        result = whisper_client.automatic_speech_recognition(audio_path)
        return result.text if hasattr(result, "text") else str(result)
    except Exception as e:
        return f"[Transcription Error: {str(e)}]"

def speak(text):
    """Voice output: turn Thunder's reply into playable audio."""
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
    """File reading: extract text from an uploaded file."""
    if file_path is None:
        return ""
    try:
        file_name = file_path.name if hasattr(file_path, "name") else file_path
        ext = os.path.splitext(file_name)[1].lower()
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_name)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return "[Error: Please add pypdf to requirements.txt]"
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
    """Live web search using DuckDuckGo (Fixed implementation)."""
    try:
        from duckduckgo_search import DDGS
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
                    "Use them if helpful:\n\n" + results
                )

        if file_context:
            full_system_prompt += (
                "\n\nThe user has shared a workspace file context. Use its contents when relevant:\n\n" + file_context
            )

        messages = [{"role": "system", "content": full_system_prompt}]
        
        # Flawlessly convert list-of-lists layout to OpenAI/HF format
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                if turn[0]: messages.append({"role": "user", "content": turn[0]})
                if turn[1]: messages.append({"role": "assistant", "content": turn[1]})

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

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"), css=custom_css) as demo:
    gr.Markdown("# ⚡ THUNDER WORKSPACE // Core v7.2")
    gr.Markdown("Production Environment. Speak, type text, scan files, or toggle real-time query engines simultaneously.")

    chatbot = gr.Chatbot(value=load_history_as_list(), height=500)

    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message here or speak into the microphone...", scale=7)
        send_btn = gr.Button("Send", scale=1)
        audio_input = gr.Audio(source="microphone", type="filepath", scale=4)

    with gr.Row():
        file_input = gr.File(label="📎 Upload a file (.pdf, .txt, .csv, .md, .json)", scale=6)
        search_toggle = gr.Checkbox(label="🔍 Search the web for this", scale=2)
        clear_btn = gr.Button("🗑️ Clear memory", scale=2)

    reply_audio = gr.Audio(label="🔊 Thunder's voice", autoplay=True)

    audio_input.change(transcribe, inputs=[audio_input], outputs=[msg])

    # Persistent State Declarations
    system_prompt = gr.State(DEFAULT_SYSTEM_PROMPT)
    temperature = gr.State(0.75)
    max_tokens = gr.State(1024)
    file_context = gr.State("")

    file_input.change(read_file, inputs=[file_input], outputs=[file_context])

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
demo.queue(concurrency_count=3).launch(server_name="0.0.0.0", server_port=port_number)
