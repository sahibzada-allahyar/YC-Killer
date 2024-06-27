import { Router, Request, Response, NextFunction } from 'express';
import { generateToken, verifyGoogleToken } from '../middlewares/auth';
import { AppError } from '../types';
import { prisma } from '../lib/prisma';
import jwt from 'jsonwebtoken';
import env from '../config/env';

const router = Router();

// Google OAuth callback
router.post('/google', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { idToken } = req.body;

    if (!idToken) {
      throw new AppError('INVALID_REQUEST', 'ID token is required', 400);
    }

    // Verify the Google ID token
    const tokenInfo = await verifyGoogleToken(idToken);

    // Create or update user in database
    const user = await prisma.user.upsert({
      where: { id: tokenInfo.sub },
      update: {
        email: tokenInfo.email,
        name: tokenInfo.name,
        updatedAt: new Date(),
      },
      create: {
        id: tokenInfo.sub,
        email: tokenInfo.email,
        name: tokenInfo.name,
        integrations: {},
        preferences: {
          timezone: 'UTC',
          language: 'en',
          notificationPreferences: {
            email: true,
            push: true,
            reminders: true,
          },
        },
      },
    });

    // Generate JWT token
    const token = generateToken(user);

    res.json({
      success: true,
      data: {
        token,
        user: {
          id: user.id,
          email: user.email,
          name: user.name,
        },
      },
    });
  } catch (error) {
    next(error);
  }
});

// Refresh token
router.post('/refresh', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { refreshToken } = req.body;

    if (!refreshToken) {
      throw new AppError('INVALID_REQUEST', 'Refresh token is required', 400);
    }

    try {
      // Verify the refresh token
      const decoded = jwt.verify(refreshToken, env.JWT_REFRESH_SECRET) as {
        userId: string;
        tokenVersion: number;
      };

      // Check if token is blacklisted or user's token version has changed
      const user = await prisma.user.findUnique({
        where: { id: decoded.userId },
        select: { id: true, email: true, tokenVersion: true },
      });

      if (!user || user.tokenVersion !== decoded.tokenVersion) {
        throw new AppError('UNAUTHORIZED', 'Invalid refresh token', 401);
      }

      // Generate new access token
      const accessToken = generateToken(user);

      // Generate new refresh token
      const newRefreshToken = jwt.sign(
        { userId: user.id, tokenVersion: user.tokenVersion },
        env.JWT_REFRESH_SECRET,
        { expiresIn: '7d' }
      );

      res.json({
        success: true,
        data: {
          accessToken,
          refreshToken: newRefreshToken,
        },
      });
    } catch (error) {
      if (error instanceof jwt.JsonWebTokenError) {
        throw new AppError('UNAUTHORIZED', 'Invalid refresh token', 401);
      }
      throw error;
    }
  } catch (error) {
    next(error);
  }
});

// Logout
router.post('/logout', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader) {
      throw new AppError('UNAUTHORIZED', 'No token provided', 401);
    }

    const [, token] = authHeader.split(' ');
    if (!token) {
      throw new AppError('UNAUTHORIZED', 'Invalid token format', 401);
    }

    try {
      // Decode token to get user ID (without verification since it might be expired)
      const decoded = jwt.decode(token) as { userId: string };

      if (!decoded?.userId) {
        throw new AppError('UNAUTHORIZED', 'Invalid token', 401);
      }

      // Increment user's token version to invalidate all existing tokens
      await prisma.user.update({
        where: { id: decoded.userId },
        data: { tokenVersion: { increment: 1 } },
      });

      res.json({
        success: true,
        data: { message: 'Successfully logged out' },
      });
    } catch (error) {
      throw error;
    }
  } catch (error) {
    next(error);
  }
});

export default router;
