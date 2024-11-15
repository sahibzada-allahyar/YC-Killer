import { Router, Request, Response, NextFunction } from 'express';
import { authMiddleware } from '../middlewares/auth';
import { AppError, AgentMessage, Conversation } from '../types';
import { LLMAgent } from '../core/agent';
import { CalendarSkill } from '../skills/calendar-skill';
import { WebSearchSkill } from '../skills/web-search-skill';
import { prisma } from '../lib/prisma';

const router = Router();

// Apply authentication middleware to all chat routes
router.use(authMiddleware);

// Initialize a new conversation
router.post('/conversations', async (req: Request, res: Response, next: NextFunction) => {
  try {
    // Create conversation in database
    const conversation = await prisma.conversation.create({
      data: {
        userId: req.user!.id,
        context: {
          skills: ['calendar', 'webSearch'],
          memory: {},
        },
        messages: [],
      },
    });

    res.json({
      success: true,
      data: conversation,
    });
  } catch (error) {
    next(error);
  }
});

// Get conversation history
router.get('/conversations/:conversationId', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { conversationId } = req.params;

    // Fetch conversation from database
    const conversation = await prisma.conversation.findUnique({
      where: { id: conversationId },
      include: { messages: true },
    });

    if (!conversation) {
      throw new AppError('NOT_FOUND', 'Conversation not found', 404);
    }

    if (conversation.userId !== req.user!.id) {
      throw new AppError('FORBIDDEN', 'Access denied', 403);
    }

    res.json({
      success: true,
      data: conversation,
    });
  } catch (error) {
    next(error);
  }
});

// Send message to agent
router.post('/conversations/:conversationId/messages', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { conversationId } = req.params;
    const { message } = req.body;

    if (!message) {
      throw new AppError('INVALID_REQUEST', 'Message content is required', 400);
    }

    // Fetch conversation from database
    const conversation = await prisma.conversation.findUnique({
      where: { id: conversationId },
      include: { messages: true },
    });

    if (!conversation) {
      throw new AppError('NOT_FOUND', 'Conversation not found', 404);
    }

    if (conversation.userId !== req.user!.id) {
      throw new AppError('FORBIDDEN', 'Access denied', 403);
    }

    // Add recent messages to context memory
    conversation.context.memory.recentMessages = conversation.messages.slice(-5);

    // Create agent instance with available skills
    const agent = new LLMAgent(req.user!, conversation.context);

    // Register available skills
    agent.registerSkill(new CalendarSkill());
    agent.registerSkill(new WebSearchSkill());

    // Process the message
    const agentMessage = await agent.processMessage(message);

    // Create user message
    const userMessage: AgentMessage = {
      role: 'user',
      content: message,
      timestamp: new Date(),
    };

    // Save messages to database
    await prisma.message.createMany({
      data: [
        {
          conversationId: conversation.id,
          role: userMessage.role,
          content: userMessage.content,
          metadata: userMessage.metadata,
        },
        {
          conversationId: conversation.id,
          role: agentMessage.role,
          content: agentMessage.content,
          metadata: agentMessage.metadata,
        },
      ],
    });

    // Update conversation timestamp
    await prisma.conversation.update({
      where: { id: conversation.id },
      data: { updatedAt: new Date() },
    });

    res.json({
      success: true,
      data: {
        message: agentMessage,
        conversation,
      },
    });
  } catch (error) {
    next(error);
  }
});

// List user's conversations
router.get('/conversations', async (req: Request, res: Response, next: NextFunction) => {
  try {
    // Fetch conversations from database with pagination
    const page = parseInt(req.query.page as string) || 1;
    const limit = parseInt(req.query.limit as string) || 10;
    const skip = (page - 1) * limit;

    const [conversations, total] = await Promise.all([
      prisma.conversation.findMany({
        where: { userId: req.user!.id },
        orderBy: { updatedAt: 'desc' },
        skip,
        take: limit,
        include: {
          messages: {
            orderBy: { createdAt: 'desc' },
            take: 1,
          },
        },
      }),
      prisma.conversation.count({
        where: { userId: req.user!.id },
      }),
    ]);

    res.json({
      success: true,
      data: {
        conversations,
        total,
        page,
        pageSize: limit,
      },
    });
  } catch (error) {
    next(error);
  }
});

// Delete conversation
router.delete('/conversations/:conversationId', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { conversationId } = req.params;

    // Delete conversation from database
    const conversation = await prisma.conversation.findUnique({
      where: { id: conversationId },
    });

    if (!conversation) {
      throw new AppError('NOT_FOUND', 'Conversation not found', 404);
    }

    if (conversation.userId !== req.user!.id) {
      throw new AppError('FORBIDDEN', 'Access denied', 403);
    }

    await prisma.conversation.delete({
      where: { id: conversationId },
    });

    res.json({
      success: true,
      data: {
        message: `Conversation ${conversationId} deleted successfully`,
      },
    });
  } catch (error) {
    next(error);
  }
});

export default router;
