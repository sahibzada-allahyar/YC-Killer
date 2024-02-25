# Agentic Accounting Firm

An intelligent accounting assistant that uses natural language processing to help small businesses manage their accounting and taxes through voice and text commands.

## Features

- Natural language processing for accounting operations
- Voice command support via OpenAI's voice agent
- Integration with QuickBooks for automated accounting
- Real-time transaction processing and categorization
- Automated tax calculation and reporting
- Budget tracking and financial insights
- Multi-currency support
- Secure authentication and data handling

## Tech Stack

- **Language:** TypeScript
- **Backend:** Node.js with Express
- **Natural Language Processing:** OpenAI GPT-4
- **Voice Processing:** OpenAI Whisper
- **Accounting Integration:** QuickBooks API
- **Database:** PostgreSQL (planned)
- **Testing:** Jest, React Testing Library
- **Documentation:** TypeDoc, OpenAPI/Swagger

## Project Structure

```
YC-Killer/
└── Agentic-Accounting-Firm/
    ├── packages/
    │   ├── shared/           # Shared types and utilities
    │   │   ├── src/
    │   │   │   ├── types/   # TypeScript type definitions
    │   │   │   └── utils/   # Shared utility functions
    │   │   └── package.json
    │   │
    │   └── backend/         # Backend server implementation
    │       ├── src/
    │       │   ├── config/  # Configuration management
    │       │   ├── routes/  # API route definitions
    │       │   ├── services/# Business logic services
    │       │   └── utils/   # Backend utilities
    │       └── package.json
    │
    ├── package.json         # Root package.json
    ├── tsconfig.json       # TypeScript configuration
    └── README.md           # Project documentation
```

## Getting Started

### Prerequisites

- Node.js >= 18.0.0
- npm >= 9.0.0
- QuickBooks Developer Account
- OpenAI API Key

### Installation

1. Navigate to the project directory:
   ```bash
   cd YC-Killer/Agentic-Accounting-Firm
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Set up environment variables:
   ```bash
   cp packages/backend/.env.example packages/backend/.env
   ```
   Edit `.env` and add your API keys and configuration.

4. Build the project:
   ```bash
   npm run build
   ```

5. Start the development server:
   ```bash
   npm run dev
   ```

### Configuration

The following environment variables are required:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_ORG_ID=your_openai_org_id

# QuickBooks Configuration
QUICKBOOKS_CLIENT_ID=your_quickbooks_client_id
QUICKBOOKS_CLIENT_SECRET=your_quickbooks_client_secret
QUICKBOOKS_ENVIRONMENT=sandbox
QUICKBOOKS_REDIRECT_URI=http://localhost:3000/callback

# Server Configuration
PORT=3000
HOST=localhost
CORS_ORIGINS=http://localhost:3000

# Logging
LOG_LEVEL=info
LOG_FORMAT=json
```

## Development

### Available Scripts

- `npm run build` - Build all packages
- `npm run dev` - Start development server
- `npm run lint` - Run ESLint
- `npm run test` - Run tests
- `npm run typecheck` - Run TypeScript type checking
- `npm run clean` - Clean build outputs
- `npm run format` - Format code with Prettier

### Code Style

This project uses:
- ESLint for code linting
- Prettier for code formatting
- TypeScript strict mode
- Conventional Commits for commit messages

### Testing

Tests are written using Jest. Run the test suite:

```bash
npm test
```

For test coverage:

```bash
npm test -- --coverage
```

## API Documentation

API documentation is generated using OpenAPI/Swagger and is available at:
- Development: http://localhost:3000/api-docs
- Production: https://your-domain.com/api-docs

## Contributing

See the main repository's [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for GPT-4 and Whisper APIs
- Intuit for QuickBooks API
- The open-source community for the amazing tools and libraries

## Support

For support, please open an issue in the main repository.
