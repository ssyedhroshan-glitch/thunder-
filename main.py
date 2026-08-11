from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

app = FastAPI(title="Thunder AI Engine")

# Enable CORS so web apps can communicate with your Render backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (can narrow down to specific domains)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

async def generate_ai_stream(prompt: str):
    words = f"Echo response to: '{prompt}' - AI engine processing complete.".split()
    for word in words:
        yield f"data: {word}\n\n"
        await asyncio.sleep(0.08)

@app.get("/")
def root():
    return {"status": "Thunder AI Engine Online"}

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        generate_ai_stream(request.prompt),
        media_type="text/event-stream"
    )
