
import sys
import os
import json
import asyncio
from typing import AsyncGenerator, Dict, Any

# Add parent directory to path to import wolfram modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from wolfram.client import WolframClient, Budget
except ImportError:
    # Fallback/Mock if running in isolation or path issue
    WolframClient = None
    Budget = None

from openai import AsyncOpenAI
import dotenv

dotenv.load_dotenv()

class PhysicsAgent:
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Initialize Wolfram Client if available
        self.wolfram_client = None
        if WolframClient and Budget:
            # Example budget
            budget = Budget(usd=1.0, latency_s=60.0, tokens=10000)
            self.wolfram_client = WolframClient(budget)

    async def process_query(self, query: str) -> AsyncGenerator[str, None]:
        """
        Process a user query, yielding SSE events.
        Events are JSON strings prefixed with 'data: ' (handled by FastAPI StreamingResponse usually, 
        but here we just yield the content or simple JSONs).
        We'll yield dictionaries that the caller will format as SSE.
        """
        
        # 1. Thought Process: Analyze if we need Wolfram
        yield json.dumps({"type": "thought", "content": "Analyzing query for physics/math content..."})
        await asyncio.sleep(0.5) # Simulated delay for UX

        needs_wolfram = False
        if "integral" in query.lower() or "calculate" in query.lower() or "solve" in query.lower():
             needs_wolfram = True
        
        context = ""

        if needs_wolfram and self.wolfram_client:
            yield json.dumps({"type": "thought", "content": "Detected calculation. Sending to Wolfram..."})
            
            try:
                # In a real app, we might extract the exact math query
                # For now, pass the whole query or a part of it
                # Run synchronous Wolfram client in a thread to avoid blocking the event loop
                wolfram_result = await asyncio.to_thread(self.wolfram_client.evaluate, query)
                result_text = wolfram_result.get("result", "No result")
                backend = wolfram_result.get("backend", "unknown")
                
                context += f"\nWolfram Result ({backend}): {result_text}\n"
                
                yield json.dumps({
                    "type": "tool_result", 
                    "tool": "Wolfram", 
                    "output": str(result_text)[:200] + "..." if len(str(result_text)) > 200 else str(result_text)
                })
                
            except Exception as e:
                 import traceback
                 tb = traceback.format_exc()
                 print("WOLFRAM FAILURE DEBUG:")
                 print(tb)
                 yield json.dumps({"type": "error", "content": f"Wolfram failed: {str(e)} | TB: {tb}"})
        
        elif needs_wolfram and not self.wolfram_client:
             yield json.dumps({"type": "thought", "content": "Wolfram client not available. Creating plan..."})

        # 2. Call LLM with context
        yield json.dumps({"type": "thought", "content": "Synthesizing answer with Physics Copilot..."})
        
        system_prompt = (
            "You are an advanced Physics Research Copilot. "
            "Use the provided context to answer the user's question. "
            "Be precise, use LaTeX for math, and keep the tone professional but helpful. "
            "If you used a tool, explain the results."
        )

        stream = await self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context: {context}\n\nUser Query: {query}"}
            ],
            stream=True
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield json.dumps({"type": "token", "content": chunk.choices[0].delta.content})

