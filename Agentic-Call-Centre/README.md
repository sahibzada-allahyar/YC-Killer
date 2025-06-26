# Agentic Call Centre

> Putting a whole call center into a data centre by completely automating it with AI agents. AI voice based sales system.

An AI-driven call center system built with TypeScript that handles both outbound sales calls and inbound customer support. The system uses Large Language Models for real-time conversations and includes features like call recording, transcription, and continuous improvement through RLHF (Reinforcement Learning with Human Feedback).

## Features

- 🤖 AI-powered conversations using GPT-4 or other LLMs
- 📞 Handles both outbound sales and inbound support calls
- 🔄 Real-time speech-to-text and text-to-speech conversion
- 📝 Call recording and transcription
- 📊 Admin dashboard for monitoring and analytics
- 🎯 Call outcome classification and tracking
- 📈 RLHF system for continuous improvement
- 🚀 Auto-scaling infrastructure
- 🔒 Secure authentication and authorization

## Tech Stack

- **Backend:**
  - Node.js with Express
  - TypeScript
  - Prisma (PostgreSQL)
  - OpenAI API
  - Twilio API
  - WebSocket for real-time updates

- **Frontend:**
  - Next.js
  - React
  - TypeScript
  - TailwindCSS
  - Chart.js for analytics

- **Infrastructure:**
  - Docker
  - Kubernetes
  - PostgreSQL
  - Nginx Ingress
  - Let's Encrypt for SSL

## Project Structure

```
YC-Killer/
└── Agentic-Call-Centre/
    ├── backend/           # Node.js backend server
    ├── frontend/         # Next.js frontend application
    └── infrastructure/   # Kubernetes configurations
```

## Prerequisites

- Node.js 18+
- PostgreSQL 15+
- Docker and Docker Compose
- Kubernetes cluster (for production)
- Twilio Account
- OpenAI API Key

## Local Development Setup

1. Navigate to the project directory:
   ```bash
   cd YC-Killer/Agentic-Call-Centre
   ```

2. Set up environment variables:
   ```bash
   # Backend
   cp backend/.env.example backend/.env
   # Frontend
   cp frontend/.env.example frontend/.env
   ```

3. Install dependencies:
   ```bash
   # Backend
   cd backend
   npm install
   
   # Frontend
   cd ../frontend
   npm install
   ```

4. Start the development environment using Docker Compose:
   ```bash
   docker-compose up
   ```

   This will start:
   - PostgreSQL database
   - Backend service on http://localhost:3000
   - Frontend service on http://localhost:3001

5. Initialize the database:
   ```bash
   cd backend
   npx prisma migrate dev
   ```

## Production Deployment

### Prerequisites
- Kubernetes cluster
- kubectl configured
- Helm installed
- Domain names configured
- SSL certificates (or cert-manager installed)

### Deployment Steps

1. Update Kubernetes configurations:
   ```bash
   # Update the domain names in ingress.yaml
   vim infrastructure/k8s/ingress.yaml
   
   # Update the secrets
   vim infrastructure/k8s/backend-deployment.yaml
   vim infrastructure/k8s/postgres-statefulset.yaml
   ```

2. Deploy PostgreSQL:
   ```bash
   kubectl apply -f infrastructure/k8s/postgres-statefulset.yaml
   ```

3. Deploy the backend:
   ```bash
   kubectl apply -f infrastructure/k8s/backend-deployment.yaml
   ```

4. Deploy the frontend:
   ```bash
   kubectl apply -f infrastructure/k8s/frontend-deployment.yaml
   ```

5. Deploy the ingress:
   ```bash
   kubectl apply -f infrastructure/k8s/ingress.yaml
   ```

## Architecture

The system consists of several key components:

1. **Telephony Integration:**
   - Handles incoming/outgoing calls via Twilio
   - Real-time audio streaming and conversion
   - Call recording management

2. **AI Conversation Engine:**
   - Processes speech-to-text input
   - Generates contextual responses using LLM
   - Manages conversation state
   - Converts responses to speech

3. **Admin Dashboard:**
   - Real-time call monitoring
   - Call analytics and metrics
   - RLHF feedback interface
   - User management

4. **Database Layer:**
   - Call records and transcripts
   - User data and authentication
   - Training data for RLHF
   - Analytics data

## Contributing

See the main repository's [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Sahibzada A - Singularity Research

## Acknowledgments

- OpenAI for GPT models
- Twilio for telephony services
- The open-source community

## Support

For support, please open an issue in the main repository.
