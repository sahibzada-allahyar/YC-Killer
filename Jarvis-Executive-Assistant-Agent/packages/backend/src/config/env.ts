import { z } from 'zod';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Environment variable schema
const envSchema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.string().transform(Number).default('3000'),
  MONGODB_URI: z.string().url(),
  JWT_SECRET: z.string().min(32),
  JWT_REFRESH_SECRET: z.string().min(32),
  OPENAI_API_KEY: z.string().min(1),
  ALLOWED_ORIGINS: z.string().transform(origins => origins.split(',')).default('http://localhost:3000'),
  LOG_LEVEL: z.enum(['error', 'warn', 'info', 'debug']).default('info'),
  // OAuth configurations
  GOOGLE_CLIENT_ID: z.string().optional(),
  GOOGLE_CLIENT_SECRET: z.string().optional(),
  // Optional API keys for various services
  SERPAPI_API_KEY: z.string().optional(),
  WEATHER_API_KEY: z.string().optional(),
});

// Parse and validate environment variables
const env = envSchema.parse({
  NODE_ENV: process.env.NODE_ENV,
  PORT: process.env.PORT,
  MONGODB_URI: process.env.MONGODB_URI,
  JWT_SECRET: process.env.JWT_SECRET,
  JWT_REFRESH_SECRET: process.env.JWT_REFRESH_SECRET,
  OPENAI_API_KEY: process.env.OPENAI_API_KEY,
  ALLOWED_ORIGINS: process.env.ALLOWED_ORIGINS,
  LOG_LEVEL: process.env.LOG_LEVEL,
  GOOGLE_CLIENT_ID: process.env.GOOGLE_CLIENT_ID,
  GOOGLE_CLIENT_SECRET: process.env.GOOGLE_CLIENT_SECRET,
  SERPAPI_API_KEY: process.env.SERPAPI_API_KEY,
  WEATHER_API_KEY: process.env.WEATHER_API_KEY,
});

export default env;
