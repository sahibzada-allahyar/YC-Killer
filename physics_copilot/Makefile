.PHONY: help dev

help:
	@echo "Available commands:"
	@echo "  make dev    - Starts the FastAPI development server with auto-reload"

dev:
	@echo "Starting Physics Copilot backend server..."
	@echo "Access at http://127.0.0.1:8000"
	@echo "API docs at http://127.0.0.1:8000/docs"
	hatch run uvicorn physics_copilot.backend.main:app --host 127.0.0.1 --port 8000 --reload