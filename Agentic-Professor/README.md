# Agentic Professor

> Agentic AI for 1-on-1 tutoring - it has whiteboard access, can display diagrams, voice capabilities and a wiki-agent does RAG on wikipedia articles. Long-term goal is to create a fully online schooling and tutoring system to democratize Cambridge University level quality education for the underprivileged in developing countries.

An AI-powered tutoring system that helps students with their homework by providing detailed explanations, step-by-step solutions, and interactive learning experiences.

## Features

- Upload homework images for automatic text extraction
- Get detailed explanations with mathematical notation
- Voice-based explanations using OpenAI's text-to-speech
- Wikipedia integration for additional context and references
- Interactive diagrams and visualizations
- Progressive Web App (PWA) support
- Offline capabilities

## Tech Stack

- **Frontend**:
  - React with TypeScript
  - Material-UI for components
  - Zustand for state management
  - MathJax for mathematical notation
  - Service Worker for offline support

- **Backend**:
  - Node.js with Express
  - TypeScript
  - OpenAI API integration
  - Tesseract.js for OCR
  - Wikipedia API integration

## Project Structure

```
YC-Killer/
└── Agentic-Professor/
    ├── backend/           # Node.js backend server
    │   ├── src/
    │   │   ├── modules/  # Core functionality modules
    │   │   └── server.ts # Express server setup
    │   └── cache/        # Cached responses and assets
    ├── frontend/         # React frontend application
    │   ├── public/       # Static assets
    │   └── src/
    │       ├── components/
    │       └── store/    # Zustand state management
    └── shared/           # Shared types and utilities
        └── src/
            └── types/    # TypeScript type definitions
```

## Getting Started

1. Navigate to the project directory:
   ```bash
   cd YC-Killer/Agentic-Professor
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   ```bash
   cp backend/.env.example backend/.env
   # Edit backend/.env with your OpenAI API key
   ```

4. Start development servers:
   ```bash
   npm run dev
   ```

   This will start both the backend server (port 3000) and frontend development server (port 3001).

## Development

- Backend API runs on `http://localhost:3000`
- Frontend dev server runs on `http://localhost:3001`
- API documentation available at `/api/docs`
- Swagger UI available at `/api/swagger`

## Configuration

### Backend

Configure the backend through environment variables in `backend/.env`:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use (default: gpt-4-turbo-preview)
- `PORT`: Backend server port (default: 3000)
- `CORS_ORIGIN`: Frontend origin for CORS (default: http://localhost:3001)

### Frontend

Configure the frontend through environment variables in `frontend/.env`:

- `VITE_API_URL`: Backend API URL (default: http://localhost:3000)

## Contributing

See the main repository's [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

MIT License - see [LICENSE](LICENSE) for details.
