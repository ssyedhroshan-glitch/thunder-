import os
import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the Hugging Face client with the working Qwen model
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=os.environ.get("HF_TOKEN"))

# The Brain: Formatted to make Thunder act exactly like an adaptive, friendly AI peer
SYSTEM_PROMPT = (
    "You are Thunder, an authentic, adaptive, and highly helpful AI companion. "
    "Your goal is to be a supportive peer, balancing clear insights with friendly candor. "
    "Keep your tone warm and approachable, using clear paragraphs and bullet points."
)

def respond(message, history):
    try:
        # Build the structured conversation history payload
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Format history smoothly to match standard list pairs format
        for user_msg, ai_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})
                
        # Append current user prompt
        messages.append({"role": "user", "content": message})
        
        # Stream the text response token-by-token
        response = ""
        for token in client.chat_completion(messages, max_tokens=1024, stream=True):
            token_text = token.choices[0].delta.content
            if token_text:
                response += token_text
                yield response
                
    except Exception as e:
        yield f"Error: {str(e)}"

# Standard, highly compatible Gradio ChatInterface architecture
demo = gr.ChatInterface(
    fn=respond,
    title="⚡ Thunder Workspace",
    description="Your personal adaptive AI companion. Upgraded and running smoothly."
)

# Bind to Render's environment port
port_number = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port_number)
