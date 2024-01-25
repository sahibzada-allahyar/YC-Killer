import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import { Server as SocketIOServer } from 'socket.io';
import { createServer } from 'http';
import { AppError } from './types';
import env from './config/env';
import { setupRoutes } from './routes';
import { setupWebSocket } from './websocket';
import { errorHandler } from './middlewares/error-handler';
import { requestLogger } from './middlewares/request-logger';
import { authMiddleware } from './middlewares/auth';

export class Server {
  private app: express.Application;
  private httpServer: ReturnType<typeof createServer>;
  private io: SocketIOServer;

  constructor() {
    this.app = express();
    this.httpServer = createServer(this.app);
    this.io = new SocketIOServer(this.httpServer, {
      cors: {
        origin: env.ALLOWED_ORIGINS,
        methods: ['GET', 'POST'],
        credentials: true,
      },
    });

    this.setupMiddleware();
    this.setupRoutes();
    this.setupWebSocket();
    this.setupErrorHandling();
  }

  private setupMiddleware(): void {
    // Basic middleware
    this.app.use(express.json());
    this.app.use(express.urlencoded({ extended: true }));
    this.app.use(cors({
      origin: env.ALLOWED_ORIGINS,
      credentials: true,
    }));

    // Custom middleware
    this.app.use(requestLogger);
    this.app.use(authMiddleware);
  }

  private setupRoutes(): void {
    // Health check endpoint
    this.app.get('/health', (_req: Request, res: Response) => {
      res.json({ status: 'ok', timestamp: new Date().toISOString() });
    });

    // API routes
    setupRoutes(this.app);
  }

  private setupWebSocket(): void {
    setupWebSocket(this.io);
  }

  private setupErrorHandling(): void {
    // Handle 404
    this.app.use((_req: Request, _res: Response, next: NextFunction) => {
      next(new AppError('NOT_FOUND', 'Resource not found', 404));
    });

    // Global error handler
    this.app.use(errorHandler);

    // Handle uncaught exceptions
    process.on('uncaughtException', (error: Error) => {
      console.error('Uncaught Exception:', error);
      process.exit(1);
    });

    // Handle unhandled promise rejections
    process.on('unhandledRejection', (reason: unknown) => {
      console.error('Unhandled Rejection:', reason);
      process.exit(1);
    });
  }

  public async start(): Promise<void> {
    try {
      // Initialize any necessary services (database, cache, etc.)
      // await initializeServices();

      const port = env.PORT;
      this.httpServer.listen(port, () => {
        console.log(`Server running on port ${port}`);
        console.log(`Environment: ${env.NODE_ENV}`);
      });
    } catch (error) {
      console.error('Failed to start server:', error);
      process.exit(1);
    }
  }

  public getApp(): express.Application {
    return this.app;
  }

  public getHttpServer(): ReturnType<typeof createServer> {
    return this.httpServer;
  }

  public getIO(): SocketIOServer {
    return this.io;
  }
}

// Export a function to create and start the server
export async function createServer(): Promise<Server> {
  const server = new Server();
  await server.start();
  return server;
}
