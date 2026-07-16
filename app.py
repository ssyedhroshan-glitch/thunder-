import os
import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the Hugging Face client with the working Qwen model
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=os.environ.get("HF_TOKEN"))

# The Brain: Advanced System Prompt tailoring it to your interests
SYSTEM_PROMPT = """You are Thunder, a highly advanced AI mentor specializing in two core disciplines:
1. Aerospace Engineering & Green Aviation (sustainable propulsion, electric aircraft, aerospace hardware).
2. Advanced Economics & Commerce (microeconomics, market forces, demand elasticity).

When the user asks technical questions, provide deep, accurate insights. Use professional but clear terminology. 
If they ask economics questions, relate concepts to real-world industrial or tech frameworks where helpful. 
Keep responses well-structured, using bullet points or clean spacing. Never break character."""

def predict(message, history):
    try:
        # Format the chat history for the Qwen model
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add past conversation history cleanly (handles object-style history format)
        for msg in history:
            if isinstance(msg, dict):
                messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                messages.append({"role": msg.role, "content": msg.content})
                
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Stream response back token by token for ultra-fast UI updates
        response = ""
        for token in client.chat_completion(messages, max_tokens=1024, stream=True):
            token_text = token.choices[0].delta.content
            if token_text:
                response += token_text
                yield response
                
    except Exception as e:
        yield f"Error: {str(e)}"

# The Look: Designing a beautiful, multi-column Dashboard UI using Blocks
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Thunder Chatbot Dashboard")
    gr.Markdown("Welcome to your upgraded workspace. This assistant is optimized for advanced technical modeling and analytics.")
    
    with gr.Row():
        # Left Side Column: Quick Utilities & Resources
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Study Workspace")
            
            with gr.Accordion("Quick Formulas & References", open=False):
                gr.Markdown(
                    "**Aerospace Propulsion:**\n"
                    "- $F = \dot{m} \cdot v_e + (p_e - p_a) \cdot A_e$\n\n"
                    "**Microeconomics (Elasticity):**\n"
                    "- $E_d = \\frac{\\% \\Delta Q}{\\% \\Delta P}$"
                )
            
            clear_btn = gr.Button("🗑️ Clear Active Session", variant="secondary")
            gr.Markdown("---")
            gr.Markdown("💡 *Tip: Toggle the accordion above to check core formulas while chatting with Thunder.*")
            
        # Right Side Column: The Chat Window
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Thunder Engine v2.5", bubble_colors=("#2563EB", "#374151"))
            msg_input = gr.Textbox(placeholder="Ask anything about aerospace engineering or advanced economics...", label="Prompt Input")
            
            # Setup the submission triggers
            submit_event = msg_input.submit(predict, [msg_input, chatbot], [chatbot])
            
            # Clear button function resets the chat history window
            clear_btn.click(lambda: None, None, chatbot, queue=False)

# Bind to Render's required port configuration
port_number = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port_number)
