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
    """Live web search using DuckDuckGo."""
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

# --- HANDLER LOGIC ---

def chat_handler(message, history, system_prompt, temperature, max_tokens, file_context, search_enabled):
    try:
        save_message("user", message)
        
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
        
        for turn in history:
            user_content = turn.get("user", {}).get("text", "") if isinstance(turn, dict) else turn[0]
            bot_content = turn.get("assistant", {}).get("text", "") if isinstance(turn, dict) else turn[1]
            if user_content: messages.append({"role": "user", "content": user_content})
            if bot_content: messages.append({"role": "assistant", "content": bot_content})

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
                
        if response:
            save_message("assistant", response)

    except Exception as e:
        yield f"Error: {str(e)}"

# --- CUSTOM UI DESIGN ---

custom_css = """
footer {visibility: hidden}
.gradio-container {background-color: #0b0f19;}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"), css=custom_css) as demo:
    gr.Markdown("# ⚡ THUNDER WORKSPACE // Core v8.1")
    gr.Markdown("Production Environment. Text processing, file parsing, web scraping, and speech nodes active.")

    system_prompt = gr.State(DEFAULT_SYSTEM_PROMPT)
    temperature = gr.State(0.75)
    max_tokens = gr.State(1024)
    file_context = gr.State("")

    with gr.Row():
        with gr.Column(scale=9):
            chat_ui = gr.ChatInterface(
                fn=chat_handler,
                additional_inputs=[system_prompt, temperature, max_tokens, file_context],
                type="messages",
                fill_height=True
            )
        with gr.Column(scale=3):
            gr.Markdown("### 📎 Workspace Tools")
            file_input = gr.File(label="Upload context file (.pdf, .txt, .csv, .md, .json)")
            search_toggle = gr.Checkbox(label="🔍 Search web on prompt", value=False)
            
            gr.Markdown("### 🔊 Audio Interface Node")
            audio_input = gr.Audio(source="microphone", type="filepath", label="Voice input mic")
            reply_audio = gr.Audio(label="Thunder Synthesized Output", autoplay=True)
            voice_trigger_btn = gr.Button("🔊 Speak Last Reply")
            
            clear_memory_btn = gr.Button("🗑️ Clear Database Memory", variant="stop")

    file_input.change(read_file, inputs=[file_input], outputs=[file_context])
    audio_input.change(transcribe, inputs=[audio_input], outputs=[chat_ui.textbox])

    def trigger_speech():
        conn = sqlite3.connect(DB_PATH)
        last_bot_reply = conn.execute("SELECT content FROM history WHERE role='assistant' ORDER BY id DESC LIMIT 1").fetchone()
        conn.close()
        if last_bot_reply:
            return speak(last_bot_reply[0])
        return None

    voice_trigger_btn.click(trigger_speech, None, reply_audio)

    def do_clear():
        clear_history()
        return None

    clear_memory_btn.click(do_clear, None, None)

port_number = int(os.environ.get("PORT", 10000))
demo.queue(concurrency_count=3).launch(server_name="0.0.0.0", server_port=port_number)
