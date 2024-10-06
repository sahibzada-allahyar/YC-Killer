// Export all types
export * from './types';

// Export constants
export * from './constants';

// Export utilities
export * from './utils';

// Export commonly used type combinations
import type { User, UserPreferences, Message, Conversation } from './types';

export interface UserWithPreferences extends User {
  preferences: Required<UserPreferences>;
}

export interface ConversationWithMessages extends Conversation {
  messages: Message[];
}

// Export validation schemas (to be used with zod)
import { z } from 'zod';
import { VALIDATION } from './constants';

export const emailSchema = z
  .string()
  .email('Invalid email address')
  .min(1, 'Email is required');

export const passwordSchema = z
  .string()
  .min(VALIDATION.MIN_PASSWORD_LENGTH, `Password must be at least ${VALIDATION.MIN_PASSWORD_LENGTH} characters`)
  .regex(/[A-Z]/, 'Password must contain at least one uppercase letter')
  .regex(/[a-z]/, 'Password must contain at least one lowercase letter')
  .regex(/[0-9]/, 'Password must contain at least one number')
  .regex(/[^A-Za-z0-9]/, 'Password must contain at least one special character');

export const languageSchema = z.enum(VALIDATION.SUPPORTED_LANGUAGES);

export const timezoneSchema = z.enum(VALIDATION.SUPPORTED_TIMEZONES);

export const userPreferencesSchema = z.object({
  timezone: timezoneSchema,
  language: languageSchema,
  notifications: z.object({
    email: z.boolean(),
    push: z.boolean(),
    reminders: z.boolean(),
  }),
});

// Export version
export const VERSION = '1.0.0';

// Export package info
export const PACKAGE_INFO = {
  name: '@jarvis-executive-assistant/shared',
  version: VERSION,
  description: 'Shared types, constants, and utilities for Jarvis Executive Assistant',
} as const;
