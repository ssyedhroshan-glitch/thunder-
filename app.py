import os
import gradio as gr
from transformers import pipeline

print("Loading Thunder's AI engine...")
# Using a small, efficient model optimized for free servers
st_ai = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

def predict(message, history):
    try:
        if isinstance(message, dict) and "text" in message:
            user_text = str(message["text"])
        elif isinstance(message, list):
            user_text = str(message[0])
        else:
            user_text = str(message)
            
        prompt = f"System: Your name is Thunder, a helpful AI assistant.\nUser: {user_text}\nAssistant:"
        
        output = st_ai(prompt, max_new_tokens=100)
        ai_response = str(output[0]['generated_text'])
        
        if "Assistant:" in ai_response:
            ai_response = ai_response.split("Assistant:")[-1].strip()
        elif ai_response.startswith(prompt):
            ai_response = ai_response[len(prompt):].strip()
            
        return ai_response
        
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.ChatInterface(
    fn=predict, 
    title="⚡ Thunder Chatbot"
)

# Render expects the app to run on port 10000 or a port assigned by the system
port = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port)

