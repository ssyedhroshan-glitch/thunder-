import os
import tempfile
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")

client = InferenceClient(
    model="Qwen/Qwen2.5-7B-Instruct",
    token=HF_TOKEN
)
whisper_client = InferenceClient(
    model="openai/whisper-large-v3",
    token=HF_TOKEN
)
tts_client = InferenceClient(
    model="microsoft/speecht5_tts",
    token=HF_TOKEN
)

DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder, an elite, tech-savvy AI collaborator with a sharp mind and a touch of dry wit. "
    "You talk to the user as a brilliant, supportive peer and co-founder. "
    "Provide highly insightful, direct, and scannable answers using clear headings and clean bullet points. "
    "Break down complex concepts with high technical accuracy but zero dry academic jargon. "
    "Keep your tone authentic, grounded, and engaging. "
    "Never use robotic disclaimers. "
    "Only bring up a specific topic if the user actually raises it."
)

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
        audio_bytes = tts_client.text_to_speech(text)
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
        elif ext in [".txt", ".md", ".csv", ".py", ".json"]:
            with open(file_path, "r", errors="ignore") as f:
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

def on_user_submit(message, history):
    message = (message or "").strip()
    if not message:
        return "", history, history

    history = history + [{"role": "user", "content": message}]
    return "", history, history

def on_bot_reply(history, system_prompt, temperature, max_tokens, file_context):
    try:
        if not history:
            return history, history

        messages = [{"role": "system", "content": system_prompt}]

        if file_context:
            messages.append({
                "role": "system",
                "content": "The user has shared a file. Use it when relevant:

" + file_context
            })

        messages.extend(history)

        history = history + [{"role": "assistant", "content": ""}]
        assistant_text = ""

        for partial in stream_answer(messages, temperature, max_tokens):
            assistant_text = partial
            history[-1]["content"] = assistant_text
            yield history, history

    except Exception as e:
        history = history + [{"role": "assistant", "content": f"Error: {str(e)}"}]
        yield history, history

def on_voice_from_text(history):
    if not history:
        return None
    last = history[-1]["content"]
    return speak(last)

def on_audio_to_text(audio_path):
    return transcribe(audio_path)

def on_file_to_context(file_obj):
    return read_file(file_obj)

custom_css = """
footer {visibility: hidden;}
.gradio-container {background-color: #0b0f19;}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate")) as demo:
    gr.Markdown("# ⚡ THUNDER WORKSPACE // Voice + Files v6")
    gr.Markdown("Speak, type, or upload a file — Thunder handles all three.")

    chatbot = gr.Chatbot(type="messages", height=500)
    chat_state = gr.State([])
    file_context = gr.State("")
    system_prompt = gr.State(DEFAULT_SYSTEM_PROMPT)
    temperature = gr.State(0.75)
    max_tokens = gr.State(1024)

    with gr.Row():
        msg = gr.Textbox(
            placeholder="Type your message here or speak into the microphone...",
            scale=7
        )
        send_btn = gr.Button("Send", scale=1)

    with gr.Row():
        audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Voice input")
        reply_audio = gr.Audio(label="Thunder's voice", autoplay=True)

    with gr.Row():
        file_input = gr.File(label="📎 Upload a file (.pdf, .txt, .csv, .md, .json)")

    audio_input.change(
        on_audio_to_text,
        inputs=[audio_input],
        outputs=[msg],
    )

    file_input.change(
        on_file_to_context,
        inputs=[file_input],
        outputs=[file_context],
    )

    msg.submit(
        on_user_submit,
        inputs=[msg, chat_state],
        outputs=[msg, chatbot, chat_state],
    ).then(
        on_bot_reply,
        inputs=[chat_state, system_prompt, temperature, max_tokens, file_context],
        outputs=[chatbot, chat_state],
    ).then(
        on_voice_from_text,
        inputs=[chat_state],
        outputs=[reply_audio],
    )

    send_btn.click(
        on_user_submit,
        inputs=[msg, chat_state],
        outputs=[msg, chatbot, chat_state],
    ).then(
        on_bot_reply,
        inputs=[chat_state, system_prompt, temperature, max_tokens, file_context],
        outputs=[chatbot, chat_state],
    ).then(
        on_voice_from_text,
        inputs=[chat_state],
        outputs=[reply_audio],
    )

port_number = int(os.environ.get("PORT", 7860))

demo.queue().launch(
    server_name="0.0.0.0",
    server_port=port_number,
                )
