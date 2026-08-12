interface ChatPayload {
  prompt: string;
  temperature?: number;
}

async function streamAiResponse(payload: ChatPayload): Promise<void> {
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "";
  const response = await fetch(`${API_BASE_URL}/api/chat/stream`, { 
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

