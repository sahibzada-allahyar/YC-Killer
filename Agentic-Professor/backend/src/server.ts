import express from 'express';
import cors from 'cors';
import bodyParser from 'body-parser';
import dotenv from 'dotenv';
import {
  ApiResponse,
  ErrorCode,
  ApiError,
  HomeworkSubmission,
  TutorResponse,
  DEFAULT_CONFIG
} from '@ai-tutor/shared';
import { TutorOrchestrator } from './modules/tutor/tutorOrchestrator';

// Load environment variables
dotenv.config();

const app = express();
const DEFAULT_PORT = 3000;

// Configure middleware
app.use(cors({
  origin: process.env.CORS_ORIGIN || 'http://localhost:3001',
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

app.use(bodyParser.json({ limit: process.env.MAX_REQUEST_SIZE || '10mb' }));

// Initialize tutor with configuration
const tutor = new TutorOrchestrator(
  process.env.OPENAI_API_KEY!,
  {
    ...DEFAULT_CONFIG,
    llm: {
      ...DEFAULT_CONFIG.llm,
      model: process.env.OPENAI_MODEL || DEFAULT_CONFIG.llm.model
    }
  }
);

// Health check endpoint
app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// Process homework endpoint
app.post('/api/process', async (req, res) => {
  try {
    const submission = req.body as HomeworkSubmission;

    // Validate request
    if (!submission) {
      throw new ApiError(
        'Missing homework submission data',
        ErrorCode.INVALID_REQUEST,
        400
      );
    }

    // Process homework
    const response = await tutor.processHomework(submission);

    // Send response
    res.json({
      success: true,
      data: response
    } as ApiResponse<TutorResponse>);
  } catch (err) {
    console.error('Error processing homework:', err);

    if (err instanceof ApiError) {
      const apiError: ApiError = err;
      res.status(apiError.statusCode).json({
        success: false,
        error: apiError
      } as ApiResponse);
    } else {
      const error = err as Error;
      const apiError = new ApiError(
        'An unexpected error occurred',
        ErrorCode.INTERNAL_ERROR,
        500,
        { message: error.message }
      );
      res.status(apiError.statusCode).json({
        success: false,
        error: apiError
      } as ApiResponse);
    }
  }
});

// Get tutor state endpoint
app.get('/api/state', (_req, res) => {
  try {
    const state = tutor.getState();
    res.json({
      success: true,
      data: state
    });
  } catch (err) {
    console.error('Error getting tutor state:', err);
    const error = err as Error;
    const apiError = new ApiError(
      'Failed to get tutor state',
      ErrorCode.INTERNAL_ERROR,
      500,
      { message: error.message }
    );
    res.status(apiError.statusCode).json({
      success: false,
      error: apiError
    } as ApiResponse);
  }
});

// Function to find an available port
const findAvailablePort = (startPort: number): Promise<number> => {
  return new Promise((resolve) => {
    const server = app.listen(startPort)
      .on('listening', () => {
        server.close(() => resolve(startPort));
      })
      .on('error', () => {
        resolve(findAvailablePort(startPort + 1));
      });
  });
};

// Start server with port finding
findAvailablePort(DEFAULT_PORT).then(port => {
  app.listen(port, () => {
    console.log(`Server running on port ${port}`);
  });
});

// Handle shutdown
process.on('SIGTERM', async () => {
  console.log('SIGTERM received. Cleaning up...');
  await tutor.cleanup();
  process.exit(0);
});

process.on('SIGINT', async () => {
  console.log('SIGINT received. Cleaning up...');
  await tutor.cleanup();
  process.exit(0);
});
