import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

interface Config {
  port: number;
  nodeEnv: string;
  twilioAccountSid: string;
  twilioAuthToken: string;
  twilioPhoneNumber: string;
  openaiApiKey: string;
  databaseUrl: string;
  jwtSecret: string;
  corsOrigin: string;
}

// Environment variables with defaults and validation
export const config: Config = {
  port: parseInt(process.env.PORT || '3000', 10),
  nodeEnv: process.env.NODE_ENV || 'development',
  twilioAccountSid: process.env.TWILIO_ACCOUNT_SID || '',
  twilioAuthToken: process.env.TWILIO_AUTH_TOKEN || '',
  twilioPhoneNumber: process.env.TWILIO_PHONE_NUMBER || '',
  openaiApiKey: process.env.OPENAI_API_KEY || '',
  databaseUrl: process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/ai_call_center',
  jwtSecret: process.env.JWT_SECRET || 'your-secret-key',
  corsOrigin: process.env.CORS_ORIGIN || 'http://localhost:3000'
};

// Validate required environment variables
const requiredEnvVars: (keyof Config)[] = [
  'twilioAccountSid',
  'twilioAuthToken',
  'twilioPhoneNumber',
  'openaiApiKey'
];

for (const envVar of requiredEnvVars) {
  if (!config[envVar]) {
    throw new Error(`Missing required environment variable: ${envVar}`);
  }
}

export default config;
