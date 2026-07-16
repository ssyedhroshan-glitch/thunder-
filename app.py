import os
import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the Hugging Face client with an incredibly smart, open model
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=os.environ.get("HF_TOKEN"))

# The Brain: Formatted to make Thunder act exactly like a supportive peer AI (like me!)
SYSTEM_PROMPT = (
    "You are Thunder, an authentic, adaptive, and deeply knowledgeable AI collaborator. "
    "Your goal is to be a helpful peer, balancing genuine empathy with clear, concise insights. "
    "Keep your tone warm, friendly, and approachable. Avoid sounding like a rigid, dry lecturer. "
    "Use clean Markdown formatting, short paragraphs, and bullet points to keep information easy to read."
)

def respond(message, history):
    try:
        # Build the structured conversation history payload
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Pull past messages safely
        for msg in history:
            if isinstance(msg, dict):
                messages.append({"role": msg["role"], "content": msg["content"]})
            elif hasattr(msg, "role") and hasattr(msg, "content"):
                messages.append({"role": msg.role, "content": msg.content})
            elif isinstance(msg, (list, tuple)) and len(msg) == 2:
                messages.append({"role": "user", "content": msg[0]})
                messages.append({"role": "assistant", "content": msg[1]})
                
        # Append current user prompt
        messages.append({"role": "user", "content": message})
        
        # Stream the text token-by-token
        response = ""
        for token in client.chat_completion(messages, max_tokens=1024, stream=True):
            token_text = token.choices[0].delta.content
            if token_text:
                response += token_text
                yield response
                
    except Exception as e:
        yield f"Error: {str(e)}"

# The Look: Corrected multi-column Blocks setup to prevent deployment crashes
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Thunder Workspace")
    gr.Markdown("Your personal adaptive AI companion.")
    
    with gr.Row():
        # Left Panel
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Controls")
            clear_btn = gr.Button("🗑️ Reset Chat", variant="secondary")
            gr.Markdown("---")
            gr.Markdown("💡 *Tip: If you want to start a completely fresh conversation, click the reset button above.*")
            
        # Right Panel (Chat Engine)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Thunder Engine v2.5", type="messages", bubble_colors=("#2563EB", "#374151"))
            
            # Use gr.ChatInterface inside Blocks for perfect stability
            gr.ChatInterface(
                fn=respond,
                chatbot=chatbot,
                clear_btn=clear_btn,
                type="messages"
            )

# Bind to Render's environment port
port_number = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port_number)

