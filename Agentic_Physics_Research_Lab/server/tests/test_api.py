
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import sys
import os

# Adjust path to find main
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "system": "Physics Copilot Online"}

@patch("agent.PhysicsAgent.process_query")
def test_chat_endpoint(mock_process_query):
    # Mock the generator
    async def mock_generator(query):
        yield '{"type": "thought", "content": "Thinking..."}'
        yield '{"type": "token", "content": "Hello"}'
    
    mock_process_query.return_value = mock_generator("test")
    
    response = client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 200
    # Streaming response verification is tricky with TestClient, usually checks headers/content type
    assert "text/event-stream" in response.headers["content-type"]
