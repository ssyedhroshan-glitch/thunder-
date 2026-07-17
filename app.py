import os
import tempfile
import gradio as gr
import urllib.request
import json
from huggingface_hub import InferenceClient

# Securely retrieve your Hugging Face token
hf_token = os.environ.get("HF_TOKEN")

# Initialize the Hugging Face clients
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", token=hf_token)
whisper_client = InferenceClient("openai/whisper-large-v3", token=hf_token)
tts_client = InferenceClient("microsoft/speecht5_tts", token=hf_token)

# The Brain: Custom peer system prompt
DEFAULT_SYSTEM_PROMPT = (
    "You are Thunder, an elite, tech-savvy AI collaborator with web-search capabilities. "
    "You talk to the user as a brilliant, supportive peer and co-founder. "
    "Provide highly insightful, direct, and scannable answers using bold headers and clean bullet points. "
    "When web context is provided, naturally synthesize it to give accurate, up-to-date real-time info. "
    "Keep your tone authentic, grounded, and engaging. Never use robotic disclaimers."
)

# --- HELPER FUNCTIONS ---

def web_search(query):
    """Perplexity Feature: Performs a quick live web search using a free engine."""
    if not query or len(query.strip()) < 3:
        return ""
    try:
        # Using a free, privacy-friendly text search API
        safe_query = urllib.parse.quote_plus(query)
        url = f"https://html.duckduckgo.com/html/?q={safe_query}"
        
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            html = response.read().decode('utf-8')
            
        # Quick fallback extraction of raw text snippets from search page
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        snippets = []
        for result in soup.find_all('a', class_='result__snippet')[:3]:
            snippets.append(result.get_text())
            
        if snippets:
            return "\n[Live Web Context Found]:\n" + "\n".join(f"- {s}" for s in snippets)
    except Exception:
        pass
    return ""

def speak(text):
    """Voice output: turn Thunder's reply into playable audio."""
    if not text:
        return None
    try:
        clean_text = text.replace("*", "").replace("#", "").replace("- ", "")
        if len(clean_text) > 250:  
            clean_text = clean_text[:250] + "..."
            
        audio_bytes = tts_client.text_to_speech(clean_text)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tmp.write(audio_bytes)
        tmp.close()
        return tmp.name
    except Exception:
        return None

def respond(message, history):
    try:
        # 1. Trigger Perplexity Web Search feature automatically for queries
        user_message = message
        if isinstance(message, dict):
            user_message = message.get("text", "")
            
        web_context = web_search(user_message)
        
        # 2. Build system prompt with live search results injected
        full_prompt = DEFAULT_SYSTEM_PROMPT
        if web_context:
            full_prompt += f"\n\n{web_context}"

        messages = [{"role": "system", "content": full_prompt}]

        # 3. Clean history management
        for turn in history:
            if isinstance(turn, (list, tuple)) and len(turn) >= 2:
                if turn[0]:
                    messages.append({"role": "user", "content": turn[0]})
                if turn[1]:
                    messages.append({"role": "assistant", "content": turn[1]})

        if isinstance(message, dict):
            files = message.get("files", [])
            if files:
                user_message += f"\n\n[User attached a file]"

        messages.append({"role": "user", "content": user_message})

        response = ""
        for token in client.chat_completion(
            messages,
            max_tokens=1024,
            temperature=0.75,
            stream=True
        ):
            token_text = token.choices[0].delta.content
            if token_text:
                response += token_text
                yield response

    except Exception as e:
        yield f"Error: {str(e)}"


# --- INTERFACE ---

custom_css = """
footer {visibility: hidden}
.gradio-container {background-color: #0b0f19;}
"""

demo = gr.ChatInterface(
    fn=respond,
    title="⚡ THUNDER WORKSPACE // Live Search v7.0",
    description="Mission Control. Voice input, File uploading, and Real-time web search combined.",
    css=custom_css,
    theme=gr.themes.Soft(primary_hue="cyan", secondary_hue="slate"),
    textbox=gr.Textbox(placeholder="Type anything (e.g., 'What is the stock price of Apple today?') or record audio...", container=True, scale=7),
)

port_number = int(os.environ.get("PORT", 10000))
demo.launch(server_name="0.0.0.0", server_port=port_number)
