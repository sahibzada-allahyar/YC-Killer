# Jarvis Executive Assistant Agent

> Making Jarvis from Iron Man a real thing.

A powerful AI-powered personal assistant that helps you manage tasks, schedule meetings, and find information efficiently. Built with TypeScript, React, and Node.js.

## Features

- 🤖 Advanced AI-powered conversation
- 📅 Calendar management and scheduling
- 📧 Email composition and management
- 🔍 Smart web search and information retrieval
- ⏰ Task and reminder management
- 🔄 Real-time updates via WebSocket
- 🔒 Secure authentication with Google OAuth
- 🎨 Modern, responsive UI with Material-UI
- 📱 Progressive Web App (PWA) support

## Tech Stack

### Frontend
- React with TypeScript
- Material-UI for components
- Zustand for state management
- Socket.io for real-time communication
- React Router for navigation
- Vite for development and building

### Backend
- Node.js with TypeScript
- Express.js for API routing
- Socket.io for WebSocket support
- MongoDB for data storage
- OpenAI API for AI capabilities
- JWT for authentication

### Shared
- TypeScript for type definitions
- Zod for validation
- Date-fns for date manipulation
- Axios for HTTP requests

## Project Structure

```
YC-Killer/
└── Jarvis-Executive-Assistant-Agent/
    ├── packages/
    │   ├── frontend/          # React frontend application
    │   ├── backend/           # Node.js backend server
    │   └── shared/            # Shared types and utilities
    ├── scripts/              # Development and deployment scripts
    ├── .vscode/             # VSCode configuration
    ├── .github/             # GitHub Actions and templates
    └── docs/                # Documentation
```

## Prerequisites

- Node.js >= 18
- MongoDB >= 5.0
- npm >= 10.2.4
- Google Cloud Platform account (for OAuth)
- OpenAI API key

## Getting Started

1. Navigate to the project directory:
   ```bash
   cd YC-Killer/Jarvis-Executive-Assistant-Agent
   ```

2. Run the setup script:
   ```bash
   ./scripts/setup.sh
   ```

3. Configure environment variables:
   - Copy `.env.example` to `.env` in the root directory
   - Update the variables with your API keys and configuration

4. Start the development servers:
   ```bash
   npm run dev
   ```

The frontend will be available at http://localhost:3000 and the backend at http://localhost:3001.

## Development

### Available Scripts

- `npm run dev`: Start development servers
- `npm run build`: Build all packages
- `npm run test`: Run tests
- `npm run lint`: Lint code
- `npm run format`: Format code with Prettier

### VSCode Setup

This project includes VSCode configuration for optimal development experience. Install the recommended extensions when prompted.

### Environment Variables

See `.env.example` for all required environment variables. Key variables include:

- `OPENAI_API_KEY`: Your OpenAI API key
- `GOOGLE_CLIENT_ID`: Google OAuth client ID
- `MONGODB_URI`: MongoDB connection string
- `JWT_SECRET`: Secret for JWT signing

## Contributing

See the main repository's [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on how to contribute to this project.

## Testing

Run tests with:

```bash
npm test                 # Run all tests
npm test -- --watch     # Run in watch mode
npm test -- --coverage  # Generate coverage report
```

## Deployment

1. Build the application:
   ```bash
   npm run build
   ```

2. Set production environment variables

3. Start the production server:
   ```bash
   npm start
   ```

See deployment guides for [Frontend](docs/frontend-deployment.md) and [Backend](docs/backend-deployment.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Sahibzada A** - *Initial work* - [Singularity Research](https://github.com/singularity-research)

## Acknowledgments

- OpenAI for GPT API
- Google Cloud Platform for OAuth integration
- All contributors and supporters

## Support

For support, please open an issue in the main repository.
