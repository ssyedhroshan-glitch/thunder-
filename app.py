import os
import gradio as gr
from huggingface_hub import InferenceClient

# We use Hugging Face's Serverless API to offload 100% of the RAM usage
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

def predict(message, history):
    try:
        # Format the chat history for a highly advanced model
        messages = [{"role": "system", "content": "Your name is Thunder, a highly intelligent and helpful AI assistant."}]
        
        # Add past conversation history
        for user_msg, ai_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})
            
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Get response from the API
        response = ""
        for token in client.chat_completion(messages, max_tokens=512, stream=True):
            token_text = token.choices[0].delta.content
            if token_text:
                response += token_text
                yield response
                
    except Exception as e:
        yield f"Error: {str(e)}"

# Define the clean Gradio Interface
demo = gr.ChatInterface(
    fn=predict, 
    title="⚡ Thunder Chatbot"
     # Use modern Gradio chat format
)

# Bind to Render's required port
port_number = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port_number)
