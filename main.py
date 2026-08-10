from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

app = FastAPI(title="Thunder AI Engine")

class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

async def generate_ai_stream(prompt: str):
    # Simulated model streaming tokens
    words = f"Echo response to: '{prompt}' - AI engine processing complete.".split()
    for word in words:
        yield f"data: {word}\n\n"
        await asyncio.sleep(0.08)

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        generate_ai_stream(request.prompt), 
        media_type="text/event-stream"
    )
    
