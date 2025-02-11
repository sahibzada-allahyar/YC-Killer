# Agentic Deep Research

An open source version of OpenAI's Deep Research. The goal behind Singularity Research is to democratize AI access as a human right and to accelerate technologies that save humanity.

## Features

- **Recursive Exploration**: Depth-first research with configurable breadth and depth
- **Parallel Processing**: Concurrent query execution with intelligent rate limiting
- **AI-Powered Analysis**: Automated insight extraction and summarization
- **Progress Tracking**: Real-time progress updates during research
- **Report Generation**: Comprehensive markdown reports with citations
- **Source Management**: Automatic source tracking and citation
- **Error Handling**: Robust error recovery and timeout management
- **Docker Support**: Containerized deployment options

## Project Structure

```
YC-Killer/
└── Agentic-Deep-Research/
    └── source/
        ├── core/                     # Core investigation logic
        │   └── comprehensive-investigation.ts
        ├── types/                    # TypeScript type definitions
        │   ├── global.d.ts          # Global type declarations
        │   ├── research.ts          # Research-related types
        │   └── search.ts            # Search operation types
        ├── utils/                    # Utility functions
        │   ├── outputmanagement.ts  # Console output handling
        │   ├── response.ts          # Response processing
        │   ├── search.ts            # Search operations
        │   └── timeline.ts          # Timeline management
        ├── providers.ts             # AI provider configuration
        ├── prompt.ts                # System prompts
        ├── run.ts                   # CLI entry point
        └── text-splitter.ts         # Text segmentation utilities
```

## Prerequisites

- Node.js >= 18.0.0
- npm or yarn
- OpenAI API key
- Firecrawl API key

## Installation

1. Navigate to the project directory:
   ```bash
   cd YC-Killer/Agentic-Deep-Research
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Create a `.env.local` file with required configuration:
   ```env
   # Required Configuration
   OPENAI_KEY=your_openai_api_key_here
   FIRECRAWL_KEY=your_firecrawl_api_key_here

   # Optional Configuration
   OPENAI_MODEL=o3-mini
   OPENAI_ENDPOINT=https://api.openai.com/v1
   FIRECRAWL_BASE_URL=https://api.firecrawl.com
   CONTEXT_SIZE=128000
   PARALLEL_EXECUTION_LIMIT=2
   MAX_DEPTH=3
   MAX_BREADTH=3
   MIN_SEGMENT_SIZE=140
   DEBUG=false
   ```

## Usage

### CLI Usage

Run a research query:

```bash
npm start "your research query here"
```

The engine will:
1. Generate strategic search queries based on your input
2. Recursively explore the topic with configurable breadth and depth
3. Extract key insights from search results
4. Generate follow-up questions for deeper investigation
5. Produce a comprehensive markdown report with citations

### Docker Usage

1. Build the container:
   ```bash
   docker build -t deep-research .
   ```

2. Run with environment variables:
   ```bash
   docker run -e OPENAI_KEY=your_key -e FIRECRAWL_KEY=your_key deep-research "your query"
   ```

Or use docker-compose:

```bash
docker-compose up
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| OPENAI_KEY | OpenAI API key | Required |
| FIRECRAWL_KEY | Firecrawl API key | Required |
| OPENAI_MODEL | AI model to use | o3-mini |
| OPENAI_ENDPOINT | Custom OpenAI endpoint | https://api.openai.com/v1 |
| FIRECRAWL_BASE_URL | Custom Firecrawl endpoint | https://api.firecrawl.com |
| CONTEXT_SIZE | Maximum context size for prompts | 128000 |
| PARALLEL_EXECUTION_LIMIT | Concurrent operation limit | 2 |
| MAX_DEPTH | Maximum exploration depth | 3 |
| MAX_BREADTH | Maximum exploration width | 3 |
| MIN_SEGMENT_SIZE | Minimum text segment size | 140 |
| DEBUG | Enable debug logging | false |

## Development

### Scripts

- Build the project:
  ```bash
  npm run build
  ```

- Run type checking:
  ```bash
  npm run lint
  ```

- Run tests:
  ```bash
  npm test
  ```

### Architecture

The engine operates through several key components:

1. **Investigation Engine** (`comprehensive-investigation.ts`):
   - Manages the recursive research process
   - Coordinates parallel query execution
   - Handles progress tracking and updates

2. **Search Operations** (`utils/search.ts`):
   - Generates strategic search queries
   - Processes and analyzes search results
   - Extracts insights and follow-up questions

3. **AI Provider** (`providers.ts`):
   - Configures AI model settings
   - Manages API interactions
   - Handles context optimization

4. **Output Management** (`utils/outputmanagement.ts`):
   - Coordinates console output
   - Manages progress display
   - Handles error reporting

### Error Handling

The system implements robust error handling:

- Timeout recovery for API calls
- Rate limit management
- Graceful degradation on partial failures
- Detailed error logging
- Progress preservation on interruption

### Debugging

Enable debug mode by setting `DEBUG=true` in your environment variables. This will:
- Log detailed operation information
- Show API request/response data
- Display memory usage statistics
- Track execution timelines

## API Documentation

### Core Functions

#### `conductInvestigation`
```typescript
async function conductInvestigation({
  query: string,
  expansionWidth: number,
  explorationDepth: number,
  insights?: string[],
  exploredSources?: string[],
  onProgress?: (progress: InvestigationProgress) => void,
}): Promise<InvestigationOutcome>
```

#### `generateAnalysisReport`
```typescript
async function generateAnalysisReport({
  initialQuery: string,
  insights: string[],
  exploredSources: string[],
}): Promise<string>
```

### Types

```typescript
interface InvestigationProgress {
  currentLevel: number;
  totalLevels: number;
  currentScope: number;
  totalScope: number;
  currentInquiry?: string;
  totalInquiries: number;
  completedInquiries: number;
}

interface InvestigationOutcome {
  insights: string[];
  exploredSources: string[];
}
```

## Contributing

See the main repository's [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

ISC

## Acknowledgments

- OpenAI for AI capabilities
- Firecrawl for knowledge retrieval
- Contributors and maintainers
