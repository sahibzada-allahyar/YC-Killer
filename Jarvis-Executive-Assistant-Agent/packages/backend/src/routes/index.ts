import { Application } from 'express';
import authRoutes from './auth.routes';
import chatRoutes from './chat.routes';
import userRoutes from './user.routes';

export function setupRoutes(app: Application): void {
  // API version prefix
  const apiPrefix = '/api/v1';

  // Mount routes
  app.use(`${apiPrefix}/auth`, authRoutes);
  app.use(`${apiPrefix}/chat`, chatRoutes);
  app.use(`${apiPrefix}/users`, userRoutes);

  // API documentation route
  app.get(`${apiPrefix}/docs`, (_req, res) => {
    res.json({
      success: true,
      data: {
        version: '1.0.0',
        description: 'Jarvis Executive Assistant API',
        endpoints: {
          auth: {
            base: `${apiPrefix}/auth`,
            routes: {
              'POST /google': 'Authenticate with Google',
              'POST /refresh': 'Refresh access token',
              'POST /logout': 'Logout user',
            },
          },
          chat: {
            base: `${apiPrefix}/chat`,
            routes: {
              'POST /conversations': 'Create new conversation',
              'GET /conversations': 'List user conversations',
              'GET /conversations/:conversationId': 'Get conversation history',
              'POST /conversations/:conversationId/messages': 'Send message to agent',
              'DELETE /conversations/:conversationId': 'Delete conversation',
            },
          },
          users: {
            base: `${apiPrefix}/users`,
            routes: {
              'GET /profile': 'Get user profile',
              'PATCH /profile': 'Update user profile',
              'PATCH /preferences': 'Update user preferences',
              'POST /integrations/:provider': 'Connect third-party integration',
              'DELETE /integrations/:provider': 'Remove third-party integration',
              'DELETE /account': 'Delete user account',
            },
          },
        },
      },
    });
  });
}
