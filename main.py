import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import AsyncOpenAI

app = FastAPI(title="Thunder AI Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Async client (reads OPENAI_API_KEY from environment variables)
client = AsyncOpenAI(api_key=api_key)

class ChatRequest(BaseModel):
    prompt: str
    temperature: float = 0.7

async def openai_stream_generator(prompt: str, temperature: float):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            stream=True,
        )

        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                # Format as Server-Sent Events (SSE)
                yield f"data: {delta.content}\n\n"

    except Exception as e:
        yield f"data: [Error: {str(e)}]\n\n"

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        openai_stream_generator(request.prompt, request.temperature),
        media_type="text/event-stream"
    )
    
