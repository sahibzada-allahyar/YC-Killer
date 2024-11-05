import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { AppError, User, UserPermission } from '../types';
import env from '../config/env';
import { prisma } from '../lib/prisma';

// Extend Express Request type to include user
declare global {
  namespace Express {
    interface Request {
      user?: User;
    }
  }
}

export const authMiddleware = async (
  req: Request,
  _res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader) {
      throw new AppError('UNAUTHORIZED', 'No authorization token provided', 401);
    }

    // Extract token from Bearer header
    const [bearer, token] = authHeader.split(' ');

    if (bearer !== 'Bearer' || !token) {
      throw new AppError('UNAUTHORIZED', 'Invalid authorization header format', 401);
    }

    try {
      // Verify and decode the JWT token
      const decoded = jwt.verify(token, env.JWT_SECRET) as {
        userId: string;
        email: string;
      };

      // Fetch complete user data from database
      const user = await prisma.user.findUnique({
        where: { id: decoded.userId },
        include: {
          preferences: true,
          integrations: true,
        },
      });

      if (!user) {
        throw new AppError('NOT_FOUND', 'User not found', 404);
      }

      // Attach user to request object
      req.user = user;
      next();
    } catch (error) {
      if (error instanceof jwt.JsonWebTokenError) {
        throw new AppError('UNAUTHORIZED', 'Invalid or expired token', 401);
      }
      throw error;
    }
  } catch (error) {
    next(error);
  }
};

// Middleware to verify specific permissions or roles
export const requirePermission = (permission: string) => {
  return async (req: Request, _res: Response, next: NextFunction): Promise<void> => {
    if (!req.user) {
      next(new AppError('UNAUTHORIZED', 'Authentication required', 401));
      return;
    }

    // Check if user has the required permission
    const userPermissions = await prisma.userPermission.findMany({
      where: { userId: req.user!.id },
      select: { permission: true },
    });

    const hasPermission = userPermissions.some((p: UserPermission) => p.permission === permission);

    if (!hasPermission) {
      next(new AppError('FORBIDDEN', `Missing required permission: ${permission}`, 403));
      return;
    }

    next();
  };
};

// Helper function to generate JWT tokens
export const generateToken = (user: User): string => {
  return jwt.sign(
    {
      userId: user.id,
      email: user.email,
    },
    env.JWT_SECRET,
    {
      expiresIn: '24h', // Token expires in 24 hours
    }
  );
};

// Helper function to verify Google OAuth tokens
export const verifyGoogleToken = async (token: string): Promise<any> => {
  try {
    const response = await fetch(
      `https://oauth2.googleapis.com/tokeninfo?id_token=${token}`
    );

    if (!response.ok) {
      throw new AppError('UNAUTHORIZED', 'Invalid Google token', 401);
    }

    return response.json();
  } catch (error) {
    throw new AppError(
      'AUTH_ERROR',
      'Failed to verify Google token',
      401,
      error instanceof Error ? error.message : undefined
    );
  }
};
