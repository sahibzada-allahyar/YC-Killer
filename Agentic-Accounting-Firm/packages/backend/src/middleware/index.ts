import { Request, Response, NextFunction } from 'express';
import rateLimit from 'express-rate-limit';
import { config } from '../config';

// Export error handling middleware
export { errorHandler, notFoundHandler, asyncHandler } from './error-handler';

// Authentication middleware
export const requireAuth = (req: Request, res: Response, next: NextFunction): void => {
  // TODO: Implement JWT authentication
  next();
};

// Role-based authorization middleware
export const requireRole = (roles: string[]) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    // TODO: Implement role-based authorization
    next();
  };
};

// Rate limiting middleware factory
export const createRateLimiter = (
  windowMs: number = 15 * 60 * 1000, // 15 minutes
  maxRequests: number = 100
) => {
  return rateLimit({
    windowMs,
    limit: maxRequests,
    message: { error: 'Too many requests from this IP, please try again later' },
    standardHeaders: true,
    legacyHeaders: false,
  });
};

// CORS configuration middleware
export const corsOptions = {
  origin: config.server.cors.origin,
  methods: config.server.cors.methods,
  credentials: true,
  maxAge: 86400, // 24 hours
};

// Request logging middleware
export const requestLogger = (req: Request, res: Response, next: NextFunction): void => {
  const start = Date.now();
  res.on('finish', () => {
    const duration = Date.now() - start;
    console.log(
      `${req.method} ${req.originalUrl} ${res.statusCode} ${duration}ms`
    );
  });
  next();
};

// Request validation middleware factory
export const validateRequest = (schema: any, source: 'body' | 'query' | 'params' = 'body') => {
  return (req: Request, res: Response, next: NextFunction): void => {
    const dataToValidate = source === 'body' ? req.body : 
                          source === 'query' ? req.query : 
                          req.params;

    const { error, value } = schema.validate(dataToValidate, {
      abortEarly: false,
      stripUnknown: true,
    });

    if (error) {
      next(error);
      return;
    }

    // Update the request with validated and transformed data
    if (source === 'body') req.body = value;
    else if (source === 'query') req.query = value;
    else req.params = value;

    next();
  };
};

// Security headers middleware
export const securityHeaders = (req: Request, res: Response, next: NextFunction): void => {
  // HSTS
  res.setHeader(
    'Strict-Transport-Security',
    'max-age=31536000; includeSubDomains'
  );

  // Content Security Policy
  res.setHeader(
    'Content-Security-Policy',
    "default-src 'self'; img-src 'self' data: https:; script-src 'self'"
  );

  // X-Content-Type-Options
  res.setHeader('X-Content-Type-Options', 'nosniff');

  // X-Frame-Options
  res.setHeader('X-Frame-Options', 'SAMEORIGIN');

  // X-XSS-Protection
  res.setHeader('X-XSS-Protection', '1; mode=block');

  // Referrer-Policy
  res.setHeader('Referrer-Policy', 'strict-origin-when-cross-origin');

  next();
};

// Error handling middleware
export const handleErrors = (err: Error, req: Request, res: Response, next: NextFunction): void => {
  console.error(err.stack);
  res.status(500).json({
    success: false,
    error: {
      message: 'Internal server error',
      ...(config.server.environment === 'development' && { stack: err.stack }),
    },
  });
};

// Compression middleware options
export const compressionOptions = {
  level: 6, // compression level
  threshold: 1024, // minimum size to compress (in bytes)
  filter: (req: Request, res: Response) => {
    if (req.headers['x-no-compression']) {
      return false;
    }
    return true;
  },
};
