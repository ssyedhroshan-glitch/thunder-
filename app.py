import os
import gradio as gr
from huggingface_hub import InferenceClient

# Initialize the Hugging Face client with the working Qwen model
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=os.environ.get("HF_TOKEN"))

# The Brain: Loaded with our adaptive peer personality!
SYSTEM_PROMPT = (
    "You are Thunder, an authentic, adaptive AI collaborator with a touch of wit. "
    "Your goal is to address the user's true intent with insightful, yet clear and concise responses. "
    "Your guiding principle is to balance empathy with candor: validate the user's thoughts and efforts "
    "authentically like a supportive, grounded peer, while correcting errors or giving feedback gently and directly. "
    "Never sound like a rigid, formal lecturer. Keep your tone warm, natural, and conversational. "
    "Use bullet points and short paragraphs to make your answers easy to scan at a glance."
)

# Step 1: Push the user's input directly into the chat history layout
def add_user_message(user_message, history):
    if not user_message.strip():
        return "", history
    # Appends the user message and sets up a blank placeholder for the bot
    return "", history + [[user_message, ""]]

# Step 2: Stream the bot's response token-by-token into that placeholder
def respond(history):
    try:
        # Build the system prompt
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Loop through all messages except the last one (which is the current, empty placeholder)
        for user_msg, ai_msg in history[:-1]:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if ai_msg:
                messages.append({"role": "assistant", "content": ai_msg})
                
        # Append the active user query (the last item in the list)
        messages.append({"role": "user", "content": history[-1][0]})
        
        # Stream the text response
        response_text = ""
        for token in client.chat_completion(messages, max_tokens=1024, stream=True):
            token_text = token.choices[0].delta.content
            if token_text:
                response_text += token_text
                # Dynamically update the bot message index in our history list
                history[-1][1] = response_text
                yield history
                
    except Exception as e:
        history[-1][1] = f"Error: {str(e)}"
        yield history

# The Look: Custom Dashboard Design using Blocks
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ⚡ Thunder Workspace")
    gr.Markdown("Your custom-designed, adaptive AI companion setup.")
    
    with gr.Row():
        # Left Panel (Controls & Sidebar info)
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Workspace Controls")
            clear_btn = gr.Button("🗑️ Clear Active Chat", variant="secondary")
            gr.Markdown("---")
            gr.Markdown(
                "### 💡 Tips & Tricks\n"
                "- **Peer Mode:** Thunder is tuned to talk like a helpful classmate, not a textbook.\n"
                "- **Need a fresh start?** Click the Clear button to wipe the board clean."
            )
            
        # Right Panel (The Chat interface)
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Thunder Engine v2.5", bubble_colors=("#2563EB", "#374151"))
            msg_input = gr.Textbox(placeholder="Type your message here and press Enter...", label="Message Input")
            
            # Chain the actions together sequentially
            msg_input.submit(
                fn=add_user_message, 
                inputs=[msg_input, chatbot], 
                outputs=[msg_input, chatbot],
                queue=False
            ).then(
                fn=respond,
                inputs=[chatbot],
                outputs=[chatbot]
            )
            
            # Click clear button resets the chat array
            clear_btn.click(lambda: [], None, chatbot, queue=False)

# Bind to Render's port
port_number = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port_number)
