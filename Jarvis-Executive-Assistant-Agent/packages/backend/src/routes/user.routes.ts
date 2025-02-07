import { Router, Request, Response, NextFunction } from 'express';
import { authMiddleware, requirePermission } from '../middlewares/auth';
import { AppError, User, UserPreferences } from '../types';
import { prisma } from '../lib/prisma';

const router = Router();

// Apply authentication middleware to all user routes
router.use(authMiddleware);

// Get user profile
router.get('/profile', async (req: Request, res: Response, next: NextFunction) => {
  try {
    // Fetch complete user profile from database with permissions
    const user = await prisma.user.findUnique({
      where: { id: req.user!.id },
      include: {
        permissions: true,
      },
    });

    if (!user) {
      throw new AppError('NOT_FOUND', 'User not found', 404);
    }

    res.json({
      success: true,
      data: {
        id: user.id,
        email: user.email,
        name: user.name,
        preferences: user.preferences,
        integrations: Object.keys(user.integrations),
        createdAt: user.createdAt,
        updatedAt: user.updatedAt,
      },
    });
  } catch (error) {
    next(error);
  }
});

// Update user profile
router.patch('/profile', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { name, preferences } = req.body;

    // Update user in database
    const updatedUser = await prisma.user.update({
      where: { id: req.user!.id },
      data: {
        name: name || undefined,
        preferences: preferences || undefined,
        updatedAt: new Date(),
      },
      include: {
        permissions: true,
      },
    });

    res.json({
      success: true,
      data: {
        id: updatedUser.id,
        email: updatedUser.email,
        name: updatedUser.name,
        preferences: updatedUser.preferences,
        integrations: Object.keys(updatedUser.integrations),
        updatedAt: updatedUser.updatedAt,
      },
    });
  } catch (error) {
    next(error);
  }
});

// Update user preferences
router.patch('/preferences', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const preferences: Partial<UserPreferences> = req.body;

    if (!preferences || Object.keys(preferences).length === 0) {
      throw new AppError('INVALID_REQUEST', 'No preferences provided', 400);
    }

    // Update preferences in database
    const updatedUser = await prisma.user.update({
      where: { id: req.user!.id },
      data: {
        preferences: {
          ...(req.user!.preferences || {
            timezone: 'UTC',
            language: 'en',
            notificationPreferences: {
              email: true,
              push: true,
              reminders: true,
            },
          }),
          ...preferences,
        },
      },
    });

    res.json({
      success: true,
      data: updatedUser.preferences,
    });
  } catch (error) {
    next(error);
  }
});

// Connect third-party integration
router.post('/integrations/:provider', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { provider } = req.params;
    const { accessToken, refreshToken, scope } = req.body;

    if (!accessToken || !refreshToken) {
      throw new AppError('INVALID_REQUEST', 'Access token and refresh token are required', 400);
    }

    // Store integration tokens securely in database
    const updatedUser = await prisma.user.update({
      where: { id: req.user!.id },
      data: {
        integrations: {
          ...(req.user!.integrations || {}),
          [provider]: {
            accessToken,
            refreshToken,
            scope: scope || [],
            expiresAt: new Date(Date.now() + 3600000), // 1 hour from now
          },
        },
      },
    });

    res.json({
      success: true,
      data: {
        provider,
        connected: true,
        scope: scope || [],
        expiresAt: new Date(Date.now() + 3600000),
      },
    });
  } catch (error) {
    next(error);
  }
});

// Remove third-party integration
router.delete('/integrations/:provider', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { provider } = req.params;

    // Remove integration from database
    const updatedUser = await prisma.user.update({
      where: { id: req.user!.id },
      data: {
        integrations: {
          ...(req.user!.integrations || {}),
          [provider]: undefined,
        },
      },
    });
    res.json({
      success: true,
      data: {
        message: `Successfully disconnected ${provider} integration`,
      },
    });
  } catch (error) {
    next(error);
  }
});

// Delete user account (requires special permission)
router.delete('/account', requirePermission('delete_account'), async (req: Request, res: Response, next: NextFunction) => {
  try {
    // Delete all user data
    await prisma.$transaction([
      // Delete all conversations and their messages (cascade)
      prisma.conversation.deleteMany({
        where: { userId: req.user!.id },
      }),
      // Delete all permissions
      prisma.userPermission.deleteMany({
        where: { userId: req.user!.id },
      }),
      // Delete the user account
      prisma.user.delete({
        where: { id: req.user!.id },
      }),
    ]);

    res.json({
      success: true,
      data: {
        message: 'Account successfully deleted',
      },
    });
  } catch (error) {
    next(error);
  }
});

export default router;
