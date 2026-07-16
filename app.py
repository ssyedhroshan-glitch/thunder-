import os
import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the Hugging Face client with the working Qwen model
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=os.environ.get("HF_TOKEN"))

# The Brain: Programmed with my exact personality traits!
SYSTEM_PROMPT = (
    "You are Thunder, an authentic, adaptive AI collaborator with a touch of wit. "
    "Your goal is to address the user's true intent with insightful, yet clear and concise responses. "
    "Your guiding principle is to balance empathy with candor: validate the user's thoughts and efforts "
    "authentically like a supportive, grounded peer, while correcting errors or giving feedback gently and directly. "
    "Never sound like a rigid, formal lecturer. Keep your tone warm, natural, and conversational. "
    "Use bullet points and short paragraphs to make your answers easy to scan at a glance."
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
