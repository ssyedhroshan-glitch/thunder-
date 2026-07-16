import os
import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the Hugging Face client with the working Qwen model
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=os.environ.get("HF_TOKEN"))

# General purpose system prompt
SYSTEM_PROMPT = "Your name is Thunder, a highly intelligent, powerful, and helpful AI assistant."

def predict(message, history):
    try:
        # Format the chat history for the model
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add past conversation history cleanly
        for msg in history:
            if isinstance(msg, dict):
                messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                messages.append({"role": msg.role, "content": msg.content})
                
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Stream response back token by token
        response = ""
        for token in client.chat_completion(messages, max_tokens=1024, stream=True):
            token_text = token.choices[0].delta.content
            if token_text:
                response += token_text
                yield response
                
    except Exception as e:
        yield f"Error: {str(e)}"

# Designing a beautiful, clean dashboard layout using gr.Blocks
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Thunder Chatbot Workspace")
    gr.Markdown("Welcome to your upgraded general-purpose AI workspace.")
    
    with gr.Row():
        # Left Side Column: Clean utilities panel
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Controls")
            clear_btn = gr.Button("🗑️ Clear Active Session", variant="secondary")
            gr.Markdown("---")
            gr.Markdown("💡 *Tip: Use the button above to instantly clear out the conversation window and reset the chat.*")
            
        # Right Side Column: The Chat Window
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Thunder Engine v2.5", bubble_colors=("#2563EB", "#374151"))
            msg_input = gr.Textbox(placeholder="Type your message here...", label="Prompt Input")
            
            # Setup the submission triggers
            submit_event = msg_input.submit(predict, [msg_input, chatbot], [chatbot])
            
            # Clear button function resets the chat history window
            clear_btn.click(lambda: None, None, chatbot, queue=False)

# Bind to Render's required port configuration
port_number = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port_number)
