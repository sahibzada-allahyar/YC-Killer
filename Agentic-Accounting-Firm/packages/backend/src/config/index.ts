type Environment = {
  [key: string]: string | undefined;
};

declare const process: {
  env: Environment;
};

export interface Config {
  openai: {
    apiKey: string;
    organization?: string;
    model: string;
    temperature: number;
    maxTokens: number;
  };
  accounting: {
    quickbooks: {
      clientId: string;
      clientSecret: string;
      refreshToken: string;
      environment: 'sandbox' | 'production';
      redirectUri: string;
    };
  };
  server: {
    port: number;
    host: string;
    environment: 'development' | 'production' | 'test';
    cors: {
      origin: string[];
      methods: string[];
    };
  };
  logging: {
    level: string;
    format: string;
  };
}

export const config: Config = {
  openai: {
    apiKey: process.env['OPENAI_API_KEY'] || '',
    organization: process.env['OPENAI_ORG_ID'],
    model: 'gpt-4',
    temperature: 0.7,
    maxTokens: 2048,
  },
  accounting: {
    quickbooks: {
      clientId: process.env['QUICKBOOKS_CLIENT_ID'] || '',
      clientSecret: process.env['QUICKBOOKS_CLIENT_SECRET'] || '',
      refreshToken: process.env['QUICKBOOKS_REFRESH_TOKEN'] || '',
      environment: (process.env['QUICKBOOKS_ENVIRONMENT'] || 'sandbox') as 'sandbox' | 'production',
      redirectUri: process.env['QUICKBOOKS_REDIRECT_URI'] || 'http://localhost:3000/callback',
    },
  },
  server: {
    port: parseInt(process.env['PORT'] || '3000', 10),
    host: process.env['HOST'] || 'localhost',
    environment: (process.env['NODE_ENV'] || 'development') as 'development' | 'production' | 'test',
    cors: {
      origin: (process.env['CORS_ORIGINS'] || 'http://localhost:3000').split(','),
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    },
  },
  logging: {
    level: process.env['LOG_LEVEL'] || 'info',
    format: process.env['LOG_FORMAT'] || 'json',
  },
};
