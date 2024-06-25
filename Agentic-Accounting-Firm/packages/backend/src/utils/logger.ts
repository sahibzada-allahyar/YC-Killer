import winston from 'winston';
import { config } from '../config';

const { format, transports } = winston;
const { combine, timestamp, printf, colorize, json } = format;

// Custom format for development logging
const devFormat = printf(({ level, message, timestamp, ...metadata }) => {
  const metaString = Object.keys(metadata).length
    ? `\n${JSON.stringify(metadata, null, 2)}`
    : '';
  return `${timestamp} [${level}]: ${message}${metaString}`;
});

// Create base logger configuration
const baseLoggerConfig = {
  level: config.logging.level,
  format: config.logging.format === 'json'
    ? combine(timestamp(), json())
    : combine(
        colorize(),
        timestamp({ format: 'YYYY-MM-DD HH:mm:ss' }),
        devFormat
      ),
  transports: [
    new transports.Console({
      handleExceptions: true,
    }),
  ],
  exitOnError: false,
};

// Create specialized loggers for different components
export const routeLogger = winston.createLogger({
  ...baseLoggerConfig,
  defaultMeta: { component: 'routes' },
});

export const commandLogger = winston.createLogger({
  ...baseLoggerConfig,
  defaultMeta: { component: 'command-processor' },
});

export const openaiLogger = winston.createLogger({
  ...baseLoggerConfig,
  defaultMeta: { component: 'openai' },
});

export const accountingLogger = winston.createLogger({
  ...baseLoggerConfig,
  defaultMeta: { component: 'accounting' },
});

export const authLogger = winston.createLogger({
  ...baseLoggerConfig,
  defaultMeta: { component: 'auth' },
});

export const quickbooksLogger = winston.createLogger({
  ...baseLoggerConfig,
  defaultMeta: { component: 'quickbooks' },
});

// Stream for Morgan HTTP request logging
export const morganStream = {
  write: (message: string) => {
    routeLogger.info(message.trim());
  },
};

// Error logging helper
export const logError = (
  logger: winston.Logger,
  message: string,
  error: unknown,
  metadata: Record<string, unknown> = {}
): void => {
  const errorInfo = error instanceof Error
    ? {
        name: error.name,
        message: error.message,
        stack: error.stack,
      }
    : error;

  logger.error(message, {
    error: errorInfo,
    ...metadata,
  });
};

// Request logging helper
export const logRequest = (
  logger: winston.Logger,
  req: {
    method: string;
    originalUrl: string;
    ip: string;
    headers: Record<string, string>;
  },
  metadata: Record<string, unknown> = {}
): void => {
  logger.info(`${req.method} ${req.originalUrl}`, {
    ip: req.ip,
    userAgent: req.headers['user-agent'],
    ...metadata,
  });
};

// Performance logging helper
export const logPerformance = (
  logger: winston.Logger,
  operation: string,
  durationMs: number,
  metadata: Record<string, unknown> = {}
): void => {
  logger.info(`${operation} completed in ${durationMs}ms`, {
    operation,
    durationMs,
    ...metadata,
  });
};

// Security logging helper
export const logSecurityEvent = (
  logger: winston.Logger,
  event: string,
  metadata: Record<string, unknown> = {}
): void => {
  logger.warn(`Security event: ${event}`, {
    event,
    timestamp: new Date().toISOString(),
    ...metadata,
  });
};

// Audit logging helper
export const logAudit = (
  logger: winston.Logger,
  action: string,
  userId: string,
  metadata: Record<string, unknown> = {}
): void => {
  logger.info(`Audit: ${action}`, {
    action,
    userId,
    timestamp: new Date().toISOString(),
    ...metadata,
  });
};

// Export default logger for general use
export default winston.createLogger({
  ...baseLoggerConfig,
  defaultMeta: { component: 'general' },
});
