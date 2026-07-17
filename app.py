import os
import tempfile
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")

# Initialize Hugging Face clients
client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)
whisper_client = InferenceClient(model="openai/whisper-large-v3", token=HF_TOKEN)
tts_client = InferenceClient(model="microsoft/speecht5_tts", token=HF_TOKEN)

DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder, an elite, tech-savvy AI collaborator with a sharp mind and a touch of dry wit. "
    "You talk to the user as a brilliant, supportive peer and co-founder. "
    "Provide highly insightful, direct, and scannable answers using clear headings and clean bullet points. "
    "Break down complex concepts with high technical accuracy but zero dry academic jargon. "
    "Keep your tone authentic, grounded, and engaging. Never use robotic disclaimers. "
    "Only bring up a specific topic if the user actually raises it."
)

# --- HELPER FUNCTIONS ---

def transcribe(audio_path):
    if not audio_path:
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
        clean_text = text.replace("*", "").replace("#", "").replace("- ", "")
        if len(clean_text) > 250:
            clean_text = clean_text[:250] + "..."
            
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
            try:
                from pypdf import PdfReader
                reader = PdfReader(file_path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return "[Error: Please add pypdf to requirements.txt]"
        elif ext in [".txt", ".md", ".csv", ".py", ".json"]:
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

def stream_answer(messages, temperature, max_tokens):
    response = ""
    for chunk in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    ):
        token_text = chunk.choices[0].delta.content or ""
        response += token_text
        yield response

# --- VERSION-AGNOSTIC STATE OPERATIONS ---

def on_user_submit(message, history):
    message = (message or "").strip()
    if not message:
        return "", history, history

    # Safely appends turn based on structure type
    if isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict):
        new_history = history + [{"role": "user", "content": message}]
    else:
        new_history = history + [[message, ""]]
        
    return "", new_history, new_history

def on_bot_reply(history, system_prompt, temperature, max_tokens, file_context):
    try:
        if not history:
            yield history, history
            return

        messages = [{"role": "system", "content": system_prompt}]

        if file_context:
            messages.append({
                "role": "system",
                "content": "The user has shared a file. Use it when relevant:\n\n" + file_context
            })

        # Dynamically inspect and parse state formats (dict vs list-of-lists)
        is_dict_format = isinstance(history[0], dict) if isinstance(history, list) and len(history) > 0 else False

        if is_dict_format:
            # Handle list of dicts format
            messages.extend(history[:-1])
            active_message = history[-1].get("content", "")
            
            history = history + [{"role": "assistant", "content": ""}]
            for partial in stream_answer(messages, temperature, max_tokens):
                history[-1]["content"] = partial
                yield history, history
        else:
            # Handle standard list-of-lists format
            for turn in history[:-1]:
                if turn[0]: messages.append({"role": "user", "content": turn[0]})
                if turn[1]: messages.append({"role": "assistant", "content": turn[1]})
            
            active_message = history[-1][0]
            messages.append({"role": "user", "content": active_message})
            
            for partial in stream_answer(messages, temperature, max_tokens):
                history[-1][1] = partial
                yield history, history

    except Exception as e:
        if isinstance(history[-1], dict):
            history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        else:
            history[-1][1] = f"Error: {str(e)}"
        yield history, history

def on_voice_from_text(history):
    if not history:
        return None
    try:
        if isinstance(history[-1], dict):
            last_reply = history[-1].get("content", "")
        else:
            last_reply = history[-1][1]
        return speak(last_reply)
    except Exception:
        return None

custom_css = """
footer {visibility: hidden;}
.gradio-container {background-color: #0b0f19;}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate")) as demo:
    gr.Markdown("# ⚡ THUNDER WORKSPACE // Voice + Files v6.2")
    gr.Markdown("Speak, type, or upload a file — Thunder handles all three features concurrently.")

    # Let Gradio default the setup structure safely
    chatbot = gr.Chatbot(height=500)
    chat_state = gr.State([])
    file_context = gr.State("")
    system_prompt = gr.State(DEFAULT_SYSTEM_PROMPT)
    temperature = gr.State(0.75)
    max_tokens = gr.State(1024)

    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message here or speak into the microphone...", scale=7)
        send_btn = gr.Button("Send", scale=1)

    with gr.Row():
        audio_input = gr.Audio(type="filepath", label="Voice input")
        reply_audio = gr.Audio(label="Thunder's voice", autoplay=True)

    with gr.Row():
        file_input = gr.File(label="📎 Upload a file (.pdf, .txt, .csv, .md, .json)")

    audio_input.change(transcribe, inputs=[audio_input], outputs=[msg])
    file_input.change(read_file, inputs=[file_input], outputs=[file_context])

    msg.submit(
        on_user_submit, inputs=[msg, chat_state], outputs=[msg, chatbot, chat_state]
    ).then(
        on_bot_reply, inputs=[chat_state, system_prompt, temperature, max_tokens, file_context], outputs=[chatbot, chat_state]
    ).then(
        on_voice_from_text, inputs=[chat_state], outputs=[reply_audio]
    )

    send_btn.click(
        on_user_submit, inputs=[msg, chat_state], outputs=[msg, chatbot, chat_state]
    ).then(
        on_bot_reply, inputs=[chat_state, system_prompt, temperature, max_tokens, file_context], outputs=[chatbot, chat_state]
    ).then(
        on_voice_from_text, inputs=[chat_state], outputs=[reply_audio]
    )

port_number = int(os.environ.get("PORT", 10000))
demo.queue().launch(server_name="0.0.0.0", server_port=port_number)
