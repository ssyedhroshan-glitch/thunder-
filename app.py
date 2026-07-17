import os
import tempfile
import gradio as gr
from huggingface_hub import InferenceClient

# Securely retrieve your Hugging Face token
hf_token = os.environ.get("HF_TOKEN")

# Initialize the Hugging Face clients
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=hf_token)
whisper_client = InferenceClient("openai/whisper-large-v3", token=hf_token)
tts_client = InferenceClient("microsoft/speecht5_tts", token=hf_token)

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

def speak(text):
    """Voice output: turn Thunder's reply into playable audio."""
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

def respond(message, history):
    try:
        # Build standard system prompt setup
        messages = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

        # Clean history iteration protecting against data layout changes
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                if turn[0]:
                    messages.append({"role": "user", "content": turn[0]})
                if turn[1]:
                    messages.append({"role": "assistant", "content": turn[1]})
            elif isinstance(turn, dict):
                role = turn.get("role")
                content = turn.get("content")
                if role and content:
                    messages.append({"role": role, "content": content})

        # Process standard input strings or files wrapped as dict inputs
        user_message = message
        if isinstance(message, dict):
            user_message = message.get("text", "")
            files = message.get("files", [])
            if files:
                user_message += f"\n\n[User uploaded a file attachment]"

        messages.append({"role": "user", "content": user_message})

        response = ""
        for token in client.chat_completion(
            messages,
            max_tokens=1024,
            temperature=0.75,
            stream=True
        ):
            token_text = token.choices[0].delta.content
            if token_text:
                response += token_text
                yield response

    except Exception as e:
        yield f"Error: {str(e)}"


# --- INTERFACE WITH STABLE WRAPPERS ---

custom_css = """
footer {visibility: hidden}
.gradio-container {background-color: #0b0f19;}
"""

# Using the native ChatInterface completely avoids structural data type mismatch crashes
demo = gr.ChatInterface(
    fn=respond,
    title="⚡ THUNDER WORKSPACE // Voice + Files v6.1",
    description="Mission Control. Speak, type text, or upload attachments natively.",
    css=custom_css,
    theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"),
    textbox=gr.Textbox(placeholder="Type your message or use your browser/device microphone button...", container=True, scale=7),
)

# Bind and launch on Render port
port_number = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port_number)
