import os
import gradio as gr
from huggingface_hub import InferenceClient

# HuggingFace Client
HF_TOKEN = os.environ.get("HF_TOKEN")
hf_client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)

# --- PDF PARSER ---
def extract_text_from_pdf(file_obj):
    if file_obj is None:
        return ""
    try:
        from pypdf import PdfReader
        file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
        reader = PdfReader(file_path)
        extracted = []
        for i, page in enumerate(reader.pages):
            if i >= 15:
                break
            text = page.extract_text()
            if text:
                extracted.append(text)
        return "\n".join(extracted)[:10000]
    except Exception as e:
        return f"[PDF Extraction Error: {str(e)}]"

# --- CHAT ENGINE ---
def chat_response(message, history, system_prompt, model_choice, pdf_file, max_tokens, temperature):
    if not message.strip():
        yield history
        return

    # Extract PDF text if present
    pdf_text = extract_text_from_pdf(pdf_file)
    full_system_prompt = system_prompt
    if pdf_text:
        full_system_prompt += f"\n\n<pdf_context>\n{pdf_text}\n</pdf_context>\nAnswer user queries using the document provided above when relevant."

    # Add user message to history
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": ""})

    # Gemini Engine
    if model_choice == "Gemini 1.5 Flash":
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_key:
            history[-1]["content"] = "⚠️ **Gemini Error:** `GEMINI_API_KEY` environment variable is missing on Render."
            yield history
            return

        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            
            contents = []
            for item in history[:-1]:
                role = "user" if item["role"] == "user" else "model"
                contents.append({"role": role, "parts": [{"text": item["content"]}]})

            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=contents,
                config={
                    "system_instruction": full_system_prompt,
                    "temperature": float(temperature),
                    "max_output_tokens": int(max_tokens)
                }
            )
            history[-1]["content"] = response.text or "[Empty Response]"
            yield history
            return
        except Exception as e:
            history[-1]["content"] = f"⚠️ **Gemini API Error:** {str(e)}"
            yield history
            return

    # Default / Qwen Engine
    messages = [{"role": "system", "content": full_system_prompt}]
    for item in history[:-1]:
        messages.append({"role": item["role"], "content": item["content"]})

    response_text = ""
    try:
        for message_chunk in hf_client.chat_completion(
            messages, max_tokens=int(max_tokens), stream=True, temperature=float(temperature)
        ):
            token = message_chunk.choices[0].delta.content
            if token:
                response_text += token
                history[-1]["content"] = response_text
                yield history
    except Exception as e:
        history[-1]["content"] = f"⚠️ **HuggingFace API Error:** {str(e)}"
        yield history

# --- GRADIO BLOCKS UI (NO CHATINTERFACE BUG) ---
with gr.Blocks(title="⚡ THUNDER WORKSPACE") as demo:
    gr.Markdown("# ⚡ THUNDER WORKSPACE")
    gr.Markdown("Multi-model Assistant with PDF document context support.")

    with gr.Row():
        with gr.Column(scale=3):
            system_prompt = gr.Textbox(
                value="You are Thunder AI, a helpful and precise collaborator.", 
                label="System Prompt", 
                lines=3
            )
            model_choice = gr.Dropdown(
                choices=["Qwen 2.5 7B (Default)", "Gemini 1.5 Flash"], 
                value="Qwen 2.5 7B (Default)", 
                label="Select Model Engine"
            )
            pdf_file = gr.File(label="Upload PDF Document", file_types=[".pdf"], file_count="single")
            max_tokens = gr.Slider(minimum=100, maximum=2048, value=512, step=64, label="Max Tokens")
            temperature = gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.05, label="Temperature")

        with gr.Column(scale=7):
            chatbot = gr.Chatbot(height=500, type="messages")
            msg = gr.Textbox(placeholder="Type your message here...", show_label=False)
            with gr.Row():
                send_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear Chat")

    # Interactive bindings
    def user_submit(user_message, history):
        return "", history

    send_btn.click(
        chat_response,
        inputs=[msg, chatbot, system_prompt, model_choice, pdf_file, max_tokens, temperature],
        outputs=[chatbot]
    ).then(
        lambda: "", None, [msg]
    )

    msg.submit(
        chat_response,
        inputs=[msg, chatbot, system_prompt, model_choice, pdf_file, max_tokens, temperature],
        outputs=[chatbot]
    ).then(
        lambda: "", None, [msg]
    )

    clear_btn.click(lambda: [], None, [chatbot])

if __name__ == "__main__":
    port_number = int(os.environ.get("PORT", 10000))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port_number,
        allowed_paths=["*"]
    )
    
