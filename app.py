import os
import gradio as gr
from huggingface_hub import InferenceClient

# Securely retrieve your Hugging Face token
hf_token = os.environ.get("HF_TOKEN")

# Initialize the Hugging Face clients
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=hf_token)
# Whisper client for transcribing voice
whisper_client = InferenceClient("openai/whisper-large-v3", token=hf_token)

# The Brain: Custom peer system prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder, an elite, tech-savvy AI collaborator with a sharp mind and a touch of dry wit. "
    "You talk to the user as a brilliant, supportive peer and co-founder. "
    "Provide highly insightful, direct, and scannable answers using bold headers and clean bullet points. "
    "Whether discussing advanced aerospace, semiconductor physics, or economics, break down complex concepts "
    "with high technical accuracy but zero dry academic jargon. "
    "Keep your tone authentic, grounded, and engaging. Never use robotic disclaimers."
)

# --- HELPER FUNCTIONS ---

def transcribe(audio_path):
    """Takes a recorded audio file path, sends it to Whisper, and returns the transcribed text."""
    if audio_path is None:
        return ""
    try:
        result = whisper_client.automatic_speech_recognition(audio_path)
        return result.text
    except Exception as e:
        return f"[Transcription Error: {str(e)}]"

def respond(message, history, system_prompt, temperature, max_tokens):
    try:
        # Build the message payload
        messages = [{"role": "system", "content": system_prompt}]

        # Robust parsing of standard history format [[user, bot], [user, bot]]
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                if turn[0]:
                    messages.append({"role": "user", "content": turn[0]})
                if turn[1]:
                    messages.append({"role": "assistant", "content": turn[1]})

        messages.append({"role": "user", "content": message})

        # Stream response
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
    gr.Markdown("# ⚡ THUNDER WORKSPACE // Voice v5.1")
    gr.Markdown("Active Mission Control. Click the Microphone to speak or use the chatbox below.")

    # Custom components
    chatbot = gr.Chatbot()

    with gr.Row():
        msg = gr.Textbox(placeholder="Type your message here or speak into the microphone...", scale=8)
        # FIXED: Switched "sources=['microphone']" to "source='microphone'" for backward compatibility
        audio_input = gr.Audio(source="microphone", type="filepath", scale=4)

    # When the user stops recording voice, translate audio to text and drop it into the textbox (msg)
    audio_input.change(transcribe, inputs=[audio_input], outputs=[msg])

    # Invisible settings inputs required by the respond function (with default values)
    system_prompt = gr.State(DEFAULT_SYSTEM_PROMPT)
    temperature = gr.State(0.75)
    max_tokens = gr.State(1024)

    # Chat submit operations
    def user_send(message, history):
        if history is None:
            history = []
        return "", history + [[message, ""]]

    def bot_reply(history, sys_prompt, temp, tokens):
        if not history:
            yield history
            return
            
        message = history[-1][0]  # Grab the last user message
        api_history = history[:-1]  # Exclude the current active turn
        
        for chunk in respond(message, api_history, sys_prompt, temp, tokens):
            history[-1][1] = chunk  # Stream content into the assistant placeholder
            yield history

    # Submit action when typing and hitting enter
    msg.submit(user_send, [msg, chatbot], [msg, chatbot]).then(
        bot_reply, [chatbot, system_prompt, temperature, max_tokens], chatbot
    )

# Bind and launch on Render port
port_number = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port_number)
