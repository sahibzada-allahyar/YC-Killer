import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import swaggerJSDoc from 'swagger-jsdoc';
import swaggerUi from 'swagger-ui-express';
import { StatusCodes } from 'http-status-codes';

import { config } from './config';
import { errorHandler, notFoundHandler } from './middleware';
import { commandRoutes } from './routes/command-routes';
import { transactionRoutes } from './routes/transaction-routes';
import { accountRoutes } from './routes/account-routes';
import { routeLogger as logger } from './utils/logger';

// Initialize Express app
const app = express();

// Security middleware
app.use(helmet());
app.use(cors({
  origin: config.server.cors.origin,
  methods: config.server.cors.methods,
  credentials: true,
}));

// Rate limiting middleware
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  limit: 100, // Limit each IP to 100 requests per windowMs
  message: { error: 'Too many requests from this IP, please try again later' },
  standardHeaders: true,
  legacyHeaders: false,
});

// Apply rate limiting to API routes
app.use('/api/', apiLimiter);

// Request parsing
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Logging
if (config.logging.format === 'simple') {
  app.use(morgan('dev'));
} else {
  app.use(morgan('combined'));
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.status(StatusCodes.OK).json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});

// API Documentation
const swaggerOptions = {
  swaggerDefinition: {
    openapi: '3.0.0',
    info: {
      title: 'AI Accounting Agent API',
      version: '1.0.0',
      description: 'API documentation for the AI Accounting Agent',
    },
    servers: [
      {
        url: `http://${config.server.host}:${config.server.port}`,
        description: 'Development server',
      },
    ],
  },
  apis: ['./src/routes/*.ts'], // Path to the API docs
};

// Generate Swagger specification
const swaggerSpec = swaggerJSDoc(swaggerOptions);

// Serve Swagger documentation
const swaggerHtml = swaggerUi.generateHTML(swaggerSpec, {
  customSiteTitle: 'AI Accounting Agent API Documentation',
  explorer: true,
});

app.use('/api-docs', swaggerUi.serveFiles(swaggerSpec));
app.get('/api-docs', (req, res) => { res.send(swaggerHtml); });

// API Routes with rate limiting
app.use('/api/commands', commandRoutes);
app.use('/api/transactions', transactionRoutes);
app.use('/api/accounts', accountRoutes);

// Error handling
app.use(notFoundHandler);
app.use(errorHandler);

// Start server
const startServer = async (): Promise<void> => {
  try {
    app.listen(config.server.port, () => {
      logger.info(
        `Server running at http://${config.server.host}:${config.server.port}`
      );
      logger.info(
        `API Documentation available at http://${config.server.host}:${config.server.port}/api-docs`
      );
    });
  } catch (error) {
    logger.error('Failed to start server', { error });
    process.exit(1);
  }
};

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  logger.error('Uncaught Exception', { error });
  process.exit(1);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection', { reason, promise });
  process.exit(1);
});

// Handle termination signals
process.on('SIGTERM', () => {
  logger.info('SIGTERM received. Shutting down gracefully...');
  process.exit(0);
});

process.on('SIGINT', () => {
  logger.info('SIGINT received. Shutting down gracefully...');
  process.exit(0);
});

// Start the server
if (require.main === module) {
  void startServer();
}

export { app };
