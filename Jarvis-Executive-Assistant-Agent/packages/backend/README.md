# Jarvis Executive Assistant - Backend

The backend service for the Jarvis Executive Assistant, providing a robust API for natural language interaction with an AI agent capable of performing various tasks through skills and integrations.

## Features

- **AI Agent Integration**: Powered by OpenAI's GPT models for natural language understanding and generation
- **Extensible Skills System**: Modular architecture for adding new capabilities
- **Real-time Communication**: WebSocket support for live chat and updates
- **OAuth Integration**: Support for Google and other third-party services
- **Secure Authentication**: JWT-based authentication and authorization
- **API Documentation**: Auto-generated API documentation endpoint

## Built With

- Node.js & TypeScript
- Express.js
- Socket.IO
- MongoDB (planned)
- OpenAI API
- Various third-party APIs (Google Calendar, SerpAPI, etc.)

## Project Structure

```
YC-Killer/
└── Jarvis-Executive-Assistant-Agent/
    └── packages/
        └── backend/
            └── src/
                ├── config/         # Configuration and environment variables
                ├── core/           # Core functionality (Agent, Base Skill)
                ├── middlewares/    # Express middlewares
                ├── routes/         # API routes
                ├── services/       # Business logic and external service integrations
                ├── skills/         # Implemented agent skills
                ├── types/          # TypeScript type definitions
                └── utils/          # Utility functions and helpers
```

## Available Skills

1. **Calendar Skill**
   - Create, read, update, and delete calendar events
   - Integrates with Google Calendar
   - Handles scheduling and reminders

2. **Web Search Skill**
   - Perform web searches using SerpAPI
   - Extract and summarize information
   - Answer user queries with up-to-date information

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- npm or yarn
- MongoDB (for production)
- API keys for required services

### Installation

1. Navigate to the project directory:
```bash
cd YC-Killer/Jarvis-Executive-Assistant-Agent/packages/backend
```

2. Install dependencies:
```bash
npm install
```

3. Create environment file:
```bash
cp .env.example .env
```

4. Update the `.env` file with your configuration and API keys.

5. Start the development server:
```bash
npm run dev
```

### Environment Variables

See `.env.example` for all required environment variables.

## API Documentation

Once the server is running, visit `/api/v1/docs` for complete API documentation.

### Main Endpoints

- **Authentication**: `/api/v1/auth/*`
  - Google OAuth
  - Token refresh
  - Logout

- **Chat**: `/api/v1/chat/*`
  - Create conversations
  - Send messages
  - Get conversation history

- **User Management**: `/api/v1/users/*`
  - Profile management
  - Preferences
  - Integration management

## WebSocket Events

- `message`: Send a message to the AI agent
- `execute_skill`: Directly execute a skill
- `agent_event`: Receive agent status updates and responses

## Development

### Adding a New Skill

1. Create a new skill class extending `BaseSkill`
2. Implement required methods and parameters
3. Register the skill in the agent initialization

Example:
```typescript
export class MyNewSkill extends BaseSkill {
  public name = 'myNewSkill';
  public description = 'Description of what the skill does';
  public parameters = [
    {
      name: 'param1',
      type: 'string',
      description: 'Parameter description',
      required: true,
    },
  ];

  public async execute(params: Record<string, unknown>, context: SkillContext): Promise<SkillResult> {
    // Implementation
  }
}
```

### Running Tests

```bash
npm run test
```

### Linting

```bash
npm run lint
```

## Contributing

See the main repository's [CONTRIBUTING.md](../../../CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Authors

- **Sahibzada A** - *Initial work* - [Singularity Research]

## Acknowledgments

- OpenAI for GPT models
- Contributors and maintainers of all used open-source packages
