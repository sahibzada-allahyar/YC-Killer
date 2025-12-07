
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
import asyncio
from agent import PhysicsAgent

app = FastAPI(title="Physics Copilot API")

# Allow CORS for local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = PhysicsAgent()

class ChatRequest(BaseModel):
    message: str

async def stream_generator(query: str):
    async for event in agent.process_query(query):
        # Format as SSE
        yield f"data: {event}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/api/chat")
async def chat(request: ChatRequest):
    return StreamingResponse(stream_generator(request.message), media_type="text/event-stream")

@app.get("/health")
def health_check():
    return {"status": "ok", "system": "Physics Copilot Online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
