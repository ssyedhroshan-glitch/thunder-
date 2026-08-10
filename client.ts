interface ChatPayload {
  prompt: string;
  temperature?: number;
}

async function streamAiResponse(payload: ChatPayload): Promise<void> {
  const response = await fetch("http://localhost:8000/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.body) return;

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    console.log("Chunk received:", chunk);
  }
}

