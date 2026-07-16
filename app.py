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
        
        # Super-robust history parsing to prevent any "unpacking" errors
        for item in history:
            # Case 1: Standard dictionary format
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role and content:
                    messages.append({"role": role, "content": content})
            
            # Case 2: Object format (Gradio ChatMessage)
            elif hasattr(item, "role") and hasattr(item, "content"):
                messages.append({"role": item.role, "content": item.content})
            
            # Case 3: List/Tuple format (supports 2-element or larger metadata tuples)
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                # We only take the first two elements (user text and assistant text)
                user_content = item[0]
                assistant_content = item[1]
                if user_content:
                    messages.append({"role": "user", "content": user_content})
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})
                
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

