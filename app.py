import os
import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the Hugging Face client
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

def respond(message, chat_history):
    if not message.strip():
        yield "", chat_history
        return

    try:
        # Build the structured conversation history payload
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Super-robust history parsing to prevent unpacking errors in custom blocks
        for item in chat_history:
            if isinstance(item, dict):
                role = item.get("role")
                content = item.get("content")
                if role and content:
                    messages.append({"role": role, "content": content})
            elif hasattr(item, "role") and hasattr(item, "content"):
                messages.append({"role": item.role, "content": item.content})
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                user_content = item[0]
                assistant_content = item[1]
                if user_content:
                    messages.append({"role": "user", "content": user_content})
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})
                
        # Append current user prompt
        messages.append({"role": "user", "content": message})
        
        # Add a placeholder for the incoming response in the UI chat history
        updated_history = list(chat_history) + [[message, ""]]
        
        # Stream the text response token-by-token
        response_text = ""
        for token in client.chat_completion(messages, max_tokens=1024, stream=True):
            token_text = token.choices[0].delta.content
            if token_text:
                response_text += token_text
                # Update the last message in history with the current stream content
                updated_history[-1][1] = response_text
                yield "", updated_history
                
    except Exception as e:
        error_history = list(chat_history) + [[message, f"Error: {str(e)}"]]
        yield "", error_history

# The Look: Designing the Custom Dashboard Workspace
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Thunder Workspace")
    gr.Markdown("Your custom-designed, adaptive AI companion setup.")
    
    with gr.Row():
        # Left Side: Control panel and tips
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Workspace Controls")
            clear_btn = gr.Button("🗑️ Clear Active Chat", variant="secondary")
            gr.Markdown("---")
            gr.Markdown(
                "### 💡 Tips & Tricks\n"
                "- **Peer Mode:** Thunder is tuned to talk like a helpful classmate, not a textbook.\n"
                "- **Need a fresh start?** Click the Clear button to wipe the board clean."
            )
            
        # Right Side: The Chat UI
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Thunder Engine v2.5", bubble_colors=("#2563EB", "#374151"))
            msg_input = gr.Textbox(placeholder="Type your message here and press Enter...", label="Message Input")
            
            # Setup actions
            # Pressing Enter submits the message
            msg_input.submit(
                fn=respond, 
                inputs=[msg_input, chatbot], 
                outputs=[msg_input, chatbot]
            )
            
            # Clicking Clear resets the chatbot window
            clear_btn.click(lambda: [], None, chatbot, queue=False)

# Bind to Render's environment port
port_number = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port_number)
