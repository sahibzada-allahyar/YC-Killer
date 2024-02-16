import { Request, Response, NextFunction } from 'express';
import { StatusCodes } from 'http-status-codes';
import type { ValidationError } from 'joi';
import { config } from '../config';

// Custom error class for API errors
export class ApiError extends Error {
  constructor(
    public statusCode: number,
    message: string,
    public details?: unknown
  ) {
    super(message);
    this.name = 'ApiError';
    Error.captureStackTrace(this, this.constructor);
  }
}

// Custom error class for business logic errors
export class BusinessError extends Error {
  constructor(
    message: string,
    public code: string,
    public details?: unknown
  ) {
    super(message);
    this.name = 'BusinessError';
    Error.captureStackTrace(this, this.constructor);
  }
}

// Error handler middleware
export const errorHandler = (
  err: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  const isDevelopment = config.server.environment === 'development';

  // Log the error
  console.error('Error:', {
    name: err.name,
    message: err.message,
    stack: isDevelopment ? err.stack : undefined,
  });

  // Handle different types of errors
  if (err instanceof ApiError) {
    res.status(err.statusCode).json({
      success: false,
      error: {
        code: err.name,
        message: err.message,
        details: err.details,
        ...(isDevelopment && { stack: err.stack }),
      },
    });
    return;
  }

  if (err instanceof BusinessError) {
    res.status(StatusCodes.BAD_REQUEST).json({
      success: false,
      error: {
        code: err.code,
        message: err.message,
        details: err.details,
        ...(isDevelopment && { stack: err.stack }),
      },
    });
    return;
  }

  // Check if error is a Joi validation error by checking for the details property
  if ('details' in err && Array.isArray((err as ValidationError).details)) {
    res.status(StatusCodes.BAD_REQUEST).json({
      success: false,
      error: {
        code: 'VALIDATION_ERROR',
        message: 'Validation failed',
        details: err.details,
        ...(isDevelopment && { stack: err.stack }),
      },
    });
    return;
  }

  // Handle unknown errors
  const statusCode =
    (err as any).statusCode || StatusCodes.INTERNAL_SERVER_ERROR;
  res.status(statusCode).json({
    success: false,
    error: {
      code: 'INTERNAL_SERVER_ERROR',
      message: isDevelopment ? err.message : 'An unexpected error occurred',
      ...(isDevelopment && { stack: err.stack }),
    },
  });
};

// Not found handler middleware
export const notFoundHandler = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  res.status(StatusCodes.NOT_FOUND).json({
    success: false,
    error: {
      code: 'NOT_FOUND',
      message: `Cannot ${req.method} ${req.path}`,
    },
  });
};

// Async handler wrapper to catch async errors
export const asyncHandler = (
  fn: (req: Request, res: Response, next: NextFunction) => Promise<any>
) => {
  return (req: Request, res: Response, next: NextFunction): void => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
};
