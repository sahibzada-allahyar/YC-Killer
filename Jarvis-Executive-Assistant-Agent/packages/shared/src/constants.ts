// API Endpoints
export const API_ENDPOINTS = {
  AUTH: {
    LOGIN: '/api/v1/auth/login',
    LOGOUT: '/api/v1/auth/logout',
    REFRESH: '/api/v1/auth/refresh',
    GOOGLE: '/api/v1/auth/google',
  },
  USERS: {
    PROFILE: '/api/v1/users/profile',
    PREFERENCES: '/api/v1/users/preferences',
    INTEGRATIONS: '/api/v1/users/integrations',
  },
  CHAT: {
    CONVERSATIONS: '/api/v1/chat/conversations',
    MESSAGES: (conversationId: string) => `/api/v1/chat/conversations/${conversationId}/messages`,
  },
} as const;

// WebSocket Events
export const WS_EVENTS = {
  CONNECT: 'connect',
  DISCONNECT: 'disconnect',
  ERROR: 'error',
  MESSAGE: 'message',
  AGENT_EVENT: 'agent_event',
} as const;

// Local Storage Keys
export const STORAGE_KEYS = {
  AUTH_TOKEN: 'auth_token',
  REFRESH_TOKEN: 'refresh_token',
  USER_PREFERENCES: 'user_preferences',
  THEME_MODE: 'theme_mode',
} as const;

// Pagination
export const DEFAULT_PAGINATION = {
  PAGE: 1,
  PAGE_SIZE: 20,
  MAX_PAGE_SIZE: 100,
} as const;

// Date Formats
export const DATE_FORMATS = {
  API: 'yyyy-MM-dd\'T\'HH:mm:ss.SSS\'Z\'',
  DISPLAY: {
    DATE: 'MMM dd, yyyy',
    TIME: 'HH:mm',
    DATE_TIME: 'MMM dd, yyyy HH:mm',
  },
} as const;

// Validation
export const VALIDATION = {
  MIN_PASSWORD_LENGTH: 8,
  MAX_MESSAGE_LENGTH: 4000,
  SUPPORTED_LANGUAGES: ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'zh', 'ja', 'ko'] as const,
  SUPPORTED_TIMEZONES: [
    'UTC',
    'America/New_York',
    'America/Los_Angeles',
    'Europe/London',
    'Europe/Paris',
    'Asia/Tokyo',
    'Asia/Shanghai',
    'Australia/Sydney',
  ] as const,
} as const;

// Error Codes
export const ERROR_CODES = {
  UNAUTHORIZED: 'UNAUTHORIZED',
  FORBIDDEN: 'FORBIDDEN',
  NOT_FOUND: 'NOT_FOUND',
  VALIDATION_ERROR: 'VALIDATION_ERROR',
  INTERNAL_ERROR: 'INTERNAL_ERROR',
  NETWORK_ERROR: 'NETWORK_ERROR',
  INTEGRATION_ERROR: 'INTEGRATION_ERROR',
} as const;

// Integration Providers
export const INTEGRATION_PROVIDERS = {
  GOOGLE: 'google',
  MICROSOFT: 'microsoft',
  SLACK: 'slack',
} as const;

// Skills
export const SKILL_CATEGORIES = {
  CALENDAR: 'calendar',
  EMAIL: 'email',
  SEARCH: 'search',
  REMINDER: 'reminder',
  WEATHER: 'weather',
  TRANSLATION: 'translation',
} as const;

// Theme
export const THEME = {
  LIGHT: 'light',
  DARK: 'dark',
  SYSTEM: 'system',
} as const;
