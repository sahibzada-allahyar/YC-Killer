// Export all types
export * from './types';

// Export utility functions (to be added as needed)
export const formatCurrency = (amount: number, currency = 'USD'): string => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency,
  }).format(amount);
};

export const formatDate = (date: string | Date, format = 'short'): string => {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return new Intl.DateTimeFormat('en-US', {
    dateStyle: format as 'short' | 'medium' | 'long' | 'full',
  }).format(dateObj);
};

export const formatNumber = (
  number: number,
  options: Intl.NumberFormatOptions = {}
): string => {
  return new Intl.NumberFormat('en-US', options).format(number);
};

// Validation helpers
export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
};

export const isValidDate = (date: string): boolean => {
  const dateObj = new Date(date);
  return dateObj instanceof Date && !isNaN(dateObj.getTime());
};

export const isValidCurrency = (currency: string): boolean => {
  const currencyRegex = /^[A-Z]{3}$/;
  return currencyRegex.test(currency);
};

// Error classes
export class ValidationError extends Error {
  constructor(message: string, public field?: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

export class BusinessError extends Error {
  constructor(message: string, public code: string) {
    super(message);
    this.name = 'BusinessError';
  }
}

// Constants
export const SUPPORTED_CURRENCIES = ['USD', 'EUR', 'GBP', 'CAD', 'AUD', 'JPY'] as const;
export const SUPPORTED_LANGUAGES = ['en', 'es', 'fr', 'de', 'it', 'ja'] as const;
export const SUPPORTED_TIMEZONES = [
  'UTC',
  'America/New_York',
  'Europe/London',
  'Asia/Tokyo',
] as const;

export const DEFAULT_PAGINATION = {
  page: 1,
  limit: 20,
} as const;

export const MAX_PAGINATION_LIMIT = 100;

// Types for constants
export type SupportedCurrency = typeof SUPPORTED_CURRENCIES[number];
export type SupportedLanguage = typeof SUPPORTED_LANGUAGES[number];
export type SupportedTimezone = typeof SUPPORTED_TIMEZONES[number];

// Type guards
export const isSupportedCurrency = (currency: string): currency is SupportedCurrency => {
  return SUPPORTED_CURRENCIES.includes(currency as SupportedCurrency);
};

export const isSupportedLanguage = (language: string): language is SupportedLanguage => {
  return SUPPORTED_LANGUAGES.includes(language as SupportedLanguage);
};

export const isSupportedTimezone = (timezone: string): timezone is SupportedTimezone => {
  return SUPPORTED_TIMEZONES.includes(timezone as SupportedTimezone);
};
