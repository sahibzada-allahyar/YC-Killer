import { Request, Response, NextFunction } from 'express';
import env from '../config/env';

export const requestLogger = (
  req: Request,
  res: Response,
  next: NextFunction
): void => {
  const start = Date.now();

  // Log request details
  const logRequest = () => {
    const duration = Date.now() - start;
    const log = {
      timestamp: new Date().toISOString(),
      method: req.method,
      path: req.path,
      query: req.query,
      statusCode: res.statusCode,
      duration: `${duration}ms`,
      userAgent: req.get('user-agent'),
      ip: req.ip,
    };

    if (env.NODE_ENV === 'development') {
      // In development, log the full request details
      console.log('Request:', JSON.stringify(log, null, 2));
    } else {
      // In production, log a condensed version
      console.log(
        `${log.timestamp} ${log.method} ${log.path} ${log.statusCode} ${log.duration}`
      );
    }
  };

  // Log after the response is sent
  res.on('finish', logRequest);
  next();
};
