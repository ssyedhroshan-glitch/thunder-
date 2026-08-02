import os
import gradio as gr
from huggingface_hub import InferenceClient

# Safe optional import for Google GenAI
HAS_GEMINI = False
genai_client = None

try:
    from google import genai
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        genai_client = genai.Client(api_key=gemini_key)
        HAS_GEMINI = True
except Exception:
    HAS_GEMINI = False

# HuggingFace Client
HF_TOKEN = os.environ.get("HF_TOKEN")
hf_client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=HF_TOKEN)

# --- PDF PROCESSING ---
def extract_text_from_pdf(file_obj):
    if file_obj is None:
        return ""
    try:
        from pypdf import PdfReader
        file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
        reader = PdfReader(file_path)
        extracted = []
        for i, page in enumerate(reader.pages):
            if i >= 15:  # Read up to first 15 pages
                break
            text = page.extract_text()
            if text:
                extracted.append(text)
        return "\n".join(extracted)[:10000] # Limit length
    except Exception as e:
        return f"[PDF Extraction Error: {str(e)}]"

# --- CHAT ENGINE ROUTER ---
def respond(message, history, system_prompt, model_choice, pdf_file, max_tokens, temperature):
    # Extract PDF context if uploaded
    pdf_text = extract_text_from_pdf(pdf_file)
    
    # Construct System Prompt with PDF Context
    full_system_prompt = system_prompt
    if pdf_text:
        full_system_prompt += f"\n\n<pdf_context>\n{pdf_text}\n</pdf_context>\nAnswer the user's queries using the document provided above when relevant."

    # Gemini Engine Execution
    if model_choice == "Gemini 1.5 Flash":
        if not HAS_GEMINI or not genai_client:
            yield "⚠️ **Gemini Error:** `GEMINI_API_KEY` environment variable is not configured on Render."
            return
        
        try:
            contents = []
            for user_msg, bot_msg in history:
                if user_msg:
                    contents.append({"role": "user", "parts": [{"text": user_msg}]})
                if bot_msg:
                    contents.append({"role": "model", "parts": [{"text": bot_msg}]})
            contents.append({"role": "user", "parts": [{"text": message}]})

            response = genai_client.models.generate_content(
                model="gemini-1.5-flash",
                contents=contents,
                config={
                    "system_instruction": full_system_prompt,
                    "temperature": float(temperature),
                    "max_output_tokens": int(max_tokens)
                }
            )
            yield response.text or "[Empty Response]"
            return
        except Exception as e:
            yield f"⚠️ **Gemini API Error:** {str(e)}"
            return

    # Fallback / Default Hugging Face Engine (Qwen 2.5 7B)
    messages = [{"role": "system", "content": full_system_prompt}]
    for user_msg, bot_msg in history:
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        if bot_msg:
            messages.append({"role": "assistant", "content": bot_msg})
    messages.append({"role": "user", "content": message})

    response = ""
    try:
        for message_chunk in hf_client.chat_completion(
            messages, max_tokens=max_tokens, stream=True, temperature=temperature
        ):
            token = message_chunk.choices[0].delta.content
            if token:
                response += token
                yield response
    except Exception as e:
        yield f"⚠️ **HuggingFace API Error:** {str(e)}"

# --- UI INTERFACE ---
available_models = ["Qwen 2.5 7B (Default)"]
if HAS_GEMINI:
    available_models.append("Gemini 1.5 Flash")

demo = gr.ChatInterface(
    respond,
    additional_inputs=[
        gr.Textbox(value="You are Thunder AI, a helpful and precise collaborator.", label="System Prompt"),
        gr.Dropdown(choices=["Qwen 2.5 7B (Default)", "Gemini 1.5 Flash"], value="Qwen 2.5 7B (Default)", label="Select Model Engine"),
        gr.File(label="Upload PDF Document", file_types=[".pdf"], file_count="single"),
        gr.Slider(minimum=100, maximum=2048, value=512, step=64, label="Max Tokens"),
        gr.Slider(minimum=0.1, maximum=1.5, value=0.7, step=0.05, label="Temperature"),
    ],
    title="⚡ THUNDER WORKSPACE",
    description="Multi-model Assistant with PDF document context support.",
)

if __name__ == "__main__":
    port_number = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port_number)
