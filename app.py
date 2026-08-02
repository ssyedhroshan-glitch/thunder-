import os
import gradio as gr
from huggingface_hub import InferenceClient

HF_TOKEN = os.environ.get("HF_TOKEN")
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)

def respond(message, history, system_message, max_tokens, temperature):
    messages = [{"role": "system", "content": system_message}]
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = ""
    for message_chunk in client.chat_completion(
        messages, max_tokens=max_tokens, stream=True, temperature=temperature
    ):
        token = message_chunk.choices[0].delta.content
        if token:
            response += token
            yield response

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are Thunder AI, a helpful assistant.", label="System Prompt"),
        gr.Slider(1, 2048, value=512, label="Max Tokens"),
        gr.Slider(0.1, 2.0, value=0.7, label="Temperature"),
    ],
    title="⚡ Thunder Workspace",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 10000)))
