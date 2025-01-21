import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { config } from '../config';
import { APIResponse } from '../utils/types';

// Extend Express Request type to include user information
declare global {
  namespace Express {
    interface Request {
      user?: {
        id: string;
        email: string;
        role: string;
      };
    }
  }
}

/**
 * Middleware to authenticate admin users using JWT
 */
export const authenticateAdmin = (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  try {
    // Get token from header
    const authHeader = req.headers.authorization;
    if (!authHeader?.startsWith('Bearer ')) {
      const response: APIResponse<null> = {
        success: false,
        error: 'No token provided'
      };
      return res.status(401).json(response);
    }

    const token = authHeader.split(' ')[1];

    // Verify token
    const decoded = jwt.verify(token, config.jwtSecret) as {
      id: string;
      email: string;
      role: string;
    };

    // Check if user has admin role
    if (decoded.role !== 'admin' && decoded.role !== 'supervisor') {
      const response: APIResponse<null> = {
        success: false,
        error: 'Unauthorized access'
      };
      return res.status(403).json(response);
    }

    // Add user info to request
    req.user = decoded;
    next();
  } catch (error) {
    console.error('Authentication error:', error);
    const response: APIResponse<null> = {
      success: false,
      error: 'Invalid token'
    };
    res.status(401).json(response);
  }
};

/**
 * Generate JWT token for authenticated users
 */
export const generateToken = (user: {
  id: string;
  email: string;
  role: string;
}): string => {
  return jwt.sign(
    {
      id: user.id,
      email: user.email,
      role: user.role
    },
    config.jwtSecret,
    { expiresIn: '24h' }
  );
};
