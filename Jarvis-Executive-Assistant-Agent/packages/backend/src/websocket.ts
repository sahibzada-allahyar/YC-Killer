import { Server as SocketIOServer, Socket } from 'socket.io';
import { verifyGoogleToken } from './middlewares/auth';
import { AgentEvent, User } from './types';
import { LLMAgent } from './core/agent';

interface AuthenticatedSocket extends Socket {
  userId?: string;
}

export function setupWebSocket(io: SocketIOServer): void {
  // Authentication middleware for WebSocket connections
  io.use(async (socket: AuthenticatedSocket, next) => {
    try {
      const token = socket.handshake.auth.token;
      if (!token) {
        throw new Error('Authentication token required');
      }

      // Verify token (you might want to use your JWT verification here instead)
      const tokenInfo = await verifyGoogleToken(token);
      socket.userId = tokenInfo.sub; // Google's user ID
      next();
    } catch (error) {
      next(new Error('Authentication failed'));
    }
  });

  io.on('connection', (socket: AuthenticatedSocket) => {
    console.log(`Client connected: ${socket.id} (User: ${socket.userId})`);

    // Join a room specific to this user
    if (socket.userId) {
      socket.join(`user:${socket.userId}`);
    }

    // Handle incoming messages
    socket.on('message', async (data: { content: string }) => {
      try {
        // Emit "thinking" event to show the agent is processing
        emitAgentEvent(io, socket.userId!, {
          type: 'thinking',
          payload: {
            conversationId: socket.id,
            messageId: Date.now().toString(),
            data: { message: 'Processing your request...' },
          },
          timestamp: new Date(),
        });

        // Create user object for the agent
        const user: User = {
          id: socket.userId!,
          email: '', // Will be populated from database in future
          integrations: {},
          createdAt: new Date(),
          updatedAt: new Date(),
        };

        // Process message through LLM Agent
        const agent = new LLMAgent(user, {
          id: socket.id,
          skills: ['calendar', 'webSearch'],
          memory: {},
        });

        const result = await agent.processMessage(data.content);

        // Emit the agent's response
        emitAgentEvent(io, socket.userId!, {
          type: 'completed',
          payload: {
            conversationId: socket.id,
            messageId: Date.now().toString(),
            data: { message: result.content },
          },
          timestamp: new Date(),
        });
      } catch (error) {
        // Emit error event
        emitAgentEvent(io, socket.userId!, {
          type: 'error',
          payload: {
            conversationId: socket.id,
            messageId: Date.now().toString(),
            data: {
              error: error instanceof Error ? error.message : 'Unknown error',
            },
          },
          timestamp: new Date(),
        });
      }
    });

    // Handle skill execution events
    socket.on('execute_skill', async (data: { skill: string; params: any }) => {
      try {
        emitAgentEvent(io, socket.userId!, {
          type: 'executing_skill',
          payload: {
            conversationId: socket.id,
            messageId: Date.now().toString(),
            data: { skill: data.skill, params: data.params },
          },
          timestamp: new Date(),
        });

        // Create user object for the agent
        const user: User = {
          id: socket.userId!,
          email: '', // Will be populated from database in future
          integrations: {},
          createdAt: new Date(),
          updatedAt: new Date(),
        };

        // Process skill execution as a message
        const agent = new LLMAgent(user, {
          id: socket.id,
          skills: ['calendar', 'webSearch'],
          memory: {},
        });

        const message = `Execute skill: ${data.skill} with parameters: ${JSON.stringify(data.params)}`;
        const result = await agent.processMessage(message);

        emitAgentEvent(io, socket.userId!, {
          type: 'completed',
          payload: {
            conversationId: socket.id,
            messageId: Date.now().toString(),
            data: { message: result.content },
          },
          timestamp: new Date(),
        });
      } catch (error) {
        emitAgentEvent(io, socket.userId!, {
          type: 'error',
          payload: {
            conversationId: socket.id,
            messageId: Date.now().toString(),
            data: {
              error: error instanceof Error ? error.message : 'Unknown error',
            },
          },
          timestamp: new Date(),
        });
      }
    });

    // Handle disconnection
    socket.on('disconnect', () => {
      console.log(`Client disconnected: ${socket.id}`);
    });
  });
}

// Helper function to emit events to a specific user
function emitAgentEvent(io: SocketIOServer, userId: string, event: AgentEvent): void {
  io.to(`user:${userId}`).emit('agent_event', event);
}
