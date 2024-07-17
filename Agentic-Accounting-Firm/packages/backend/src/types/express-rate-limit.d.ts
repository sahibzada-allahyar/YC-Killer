declare module 'express-rate-limit' {
  import { Request, Response, NextFunction, RequestHandler } from 'express';

  interface Options {
    windowMs?: number;
    limit?: number;
    message?: string | object;
    statusCode?: number;
    headers?: boolean;
    draft_polli_ratelimit_headers?: boolean;
    skipFailedRequests?: boolean;
    skipSuccessfulRequests?: boolean;
    requestWasSuccessful?: (req: Request, res: Response) => boolean;
    skip?: (req: Request, res: Response) => boolean | Promise<boolean>;
    keyGenerator?: (req: Request, res: Response) => string;
    handler?: (req: Request, res: Response, next: NextFunction) => void;
    onLimitReached?: (req: Request, res: Response, optionsUsed: Options) => void;
    standardHeaders?: boolean;
    legacyHeaders?: boolean;
    store?: any;
  }

  interface RateLimit {
    (options?: Options): RequestHandler;
    resetKey?: (key: string) => void;
    getKey?: (key: string) => any;
  }

  const rateLimit: RateLimit;
  export = rateLimit;
}
