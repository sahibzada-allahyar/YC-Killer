import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import { createServer } from 'http';
import { WebSocket, Server as WebSocketServer } from 'ws';
import { config } from './config';
import { APIResponse, WebSocketEvent } from './utils/types';
import callRoutes from './routes/callRoutes';

// Import services
import { telephonyService } from './services/telephonyService';
import { llmService } from './services/llmService';

// Create Express app
const app = express();

// Create HTTP server
const server = createServer(app);

// Create WebSocket server
const wss = new WebSocketServer({ server });

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cors({
  origin: config.corsOrigin,
  credentials: true
}));

// Mount routes
app.use('/api/call', callRoutes);

// Global error handler
app.use((err: Error, req: Request, res: Response, next: NextFunction) => {
  console.error('Unhandled Error:', err);
  const response: APIResponse<null> = {
    success: false,
    error: config.nodeEnv === 'production' 
      ? 'An unexpected error occurred' 
      : err.message
  };
  res.status(500).json(response);
});

// Handle 404s
app.use((req: Request, res: Response) => {
  const response: APIResponse<null> = {
    success: false,
    error: 'Route not found'
  };
  res.status(404).json(response);
});

// WebSocket connection handling
wss.on('connection', (ws: WebSocket) => {
  console.log('New WebSocket connection established');

  // Send initial active calls data
  const activeCalls = telephonyService.getActiveCalls();
  ws.send(JSON.stringify({
    type: 'activeCallsUpdate',
    data: activeCalls
  }));

  ws.on('message', (message: Buffer) => {
    try {
      const event: WebSocketEvent = JSON.parse(message.toString());
      console.log('Received event:', event);
      
      switch (event.type) {
        case 'callStarted':
          // Broadcast new call to all admin clients
          wss.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
              client.send(JSON.stringify({
                type: 'callStarted',
                callId: event.callId,
                data: event.data,
                timestamp: new Date()
              }));
            }
          });
          break;

        case 'transcriptUpdate':
          // Send transcript update to admin clients
          wss.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
              client.send(JSON.stringify({
                type: 'transcriptUpdate',
                callId: event.callId,
                data: event.data,
                timestamp: new Date()
              }));
            }
          });
          break;

        case 'callEnded':
          // Notify admins of call completion
          wss.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
              client.send(JSON.stringify({
                type: 'callEnded',
                callId: event.callId,
                data: event.data,
                timestamp: new Date()
              }));
            }
          });
          break;

        case 'feedbackSubmitted':
          // Broadcast feedback to relevant admin clients
          wss.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
              client.send(JSON.stringify({
                type: 'feedbackSubmitted',
                callId: event.callId,
                data: event.data,
                timestamp: new Date()
              }));
            }
          });
          break;

        default:
          console.warn('Unknown event type:', event.type);
      }
    } catch (error) {
      console.error('Error processing WebSocket message:', error);
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected');
  });

  ws.on('error', (error: Error) => {
    console.error('WebSocket error:', error);
  });

  // Ping to keep connection alive
  const pingInterval = setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.ping();
    }
  }, 30000);

  ws.on('close', () => {
    clearInterval(pingInterval);
  });
});

// Health check endpoint
app.get('/health', (req: Request, res: Response) => {
  const response: APIResponse<{ status: string; version: string }> = {
    success: true,
    data: {
      status: 'healthy',
      version: '1.0.0'
    }
  };
  res.json(response);
});

// Start server
const PORT = config.port;
server.listen(PORT, () => {
  console.log(`Server running in ${config.nodeEnv} mode on port ${PORT}`);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (err: Error) => {
  console.error('Unhandled Promise Rejection:', err);
  // In production, you might want to exit and let the process manager restart
  if (config.nodeEnv === 'production') {
    process.exit(1);
  }
});

export { app, server, wss };
