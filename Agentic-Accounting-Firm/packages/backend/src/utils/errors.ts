import { StatusCodes } from 'http-status-codes';

/**
 * Base error class for application errors
 */
export class AppError extends Error {
  constructor(
    public statusCode: number,
    message: string,
    public code: string,
    public details?: unknown
  ) {
    super(message);
    this.name = 'AppError';
    Error.captureStackTrace(this, this.constructor);
  }
}

/**
 * Error class for validation errors
 */
export class ValidationError extends AppError {
  constructor(message: string, details?: unknown) {
    super(
      StatusCodes.BAD_REQUEST,
      message,
      'VALIDATION_ERROR',
      details
    );
    this.name = 'ValidationError';
  }
}

/**
 * Error class for authentication errors
 */
export class AuthenticationError extends AppError {
  constructor(message: string, details?: unknown) {
    super(
      StatusCodes.UNAUTHORIZED,
      message,
      'AUTHENTICATION_ERROR',
      details
    );
    this.name = 'AuthenticationError';
  }
}

/**
 * Error class for authorization errors
 */
export class AuthorizationError extends AppError {
  constructor(message: string, details?: unknown) {
    super(
      StatusCodes.FORBIDDEN,
      message,
      'AUTHORIZATION_ERROR',
      details
    );
    this.name = 'AuthorizationError';
  }
}

/**
 * Error class for not found errors
 */
export class NotFoundError extends AppError {
  constructor(message: string, details?: unknown) {
    super(
      StatusCodes.NOT_FOUND,
      message,
      'NOT_FOUND_ERROR',
      details
    );
    this.name = 'NotFoundError';
  }
}

/**
 * Error class for conflict errors
 */
export class ConflictError extends AppError {
  constructor(message: string, details?: unknown) {
    super(
      StatusCodes.CONFLICT,
      message,
      'CONFLICT_ERROR',
      details
    );
    this.name = 'ConflictError';
  }
}

/**
 * Error class for rate limit errors
 */
export class RateLimitError extends AppError {
  constructor(message: string, details?: unknown) {
    super(
      StatusCodes.TOO_MANY_REQUESTS,
      message,
      'RATE_LIMIT_ERROR',
      details
    );
    this.name = 'RateLimitError';
  }
}

/**
 * Error class for external service errors
 */
export class ExternalServiceError extends AppError {
  constructor(message: string, details?: unknown) {
    super(
      StatusCodes.BAD_GATEWAY,
      message,
      'EXTERNAL_SERVICE_ERROR',
      details
    );
    this.name = 'ExternalServiceError';
  }
}

/**
 * Error class for QuickBooks API errors
 */
export interface ServiceErrorDetails {
  service: string;
  [key: string]: unknown;
}

export class QuickBooksError extends ExternalServiceError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, {
      service: 'QuickBooks',
      ...(details || {}),
    } as ServiceErrorDetails);
    this.name = 'QuickBooksError';
  }
}

/**
 * Error class for OpenAI API errors
 */
export class OpenAIError extends ExternalServiceError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, {
      service: 'OpenAI',
      ...(details || {}),
    } as ServiceErrorDetails);
    this.name = 'OpenAIError';
  }
}

/**
 * Error class for business logic errors
 */
export class BusinessError extends AppError {
  constructor(message: string, code: string, details?: unknown) {
    super(
      StatusCodes.BAD_REQUEST,
      message,
      code,
      details
    );
    this.name = 'BusinessError';
  }
}

/**
 * Helper function to determine if an error is an instance of AppError
 */
export function isAppError(error: unknown): error is AppError {
  return error instanceof AppError;
}

/**
 * Helper function to convert unknown errors to AppError
 */
export function toAppError(error: unknown): AppError {
  if (isAppError(error)) {
    return error;
  }

  if (error instanceof Error) {
    return new AppError(
      StatusCodes.INTERNAL_SERVER_ERROR,
      error.message,
      'INTERNAL_SERVER_ERROR',
      { originalError: error }
    );
  }

  return new AppError(
    StatusCodes.INTERNAL_SERVER_ERROR,
    'An unexpected error occurred',
    'INTERNAL_SERVER_ERROR',
    { originalError: error }
  );
}

/**
 * Helper function to assert that a condition is true
 */
export function assert(condition: boolean, message: string): asserts condition {
  if (!condition) {
    throw new ValidationError(message);
  }
}

/**
 * Helper function to assert that a value is not null or undefined
 */
export function assertExists<T>(
  value: T | null | undefined,
  message: string
): asserts value is T {
  if (value === null || value === undefined) {
    throw new NotFoundError(message);
  }
}

/**
 * Helper function to assert that a user is authenticated
 */
export function assertAuthenticated(
  isAuthenticated: boolean,
  message = 'Authentication required'
): asserts isAuthenticated {
  if (!isAuthenticated) {
    throw new AuthenticationError(message);
  }
}

/**
 * Helper function to assert that a user is authorized
 */
export function assertAuthorized(
  isAuthorized: boolean,
  message = 'Not authorized'
): asserts isAuthorized {
  if (!isAuthorized) {
    throw new AuthorizationError(message);
  }
}
