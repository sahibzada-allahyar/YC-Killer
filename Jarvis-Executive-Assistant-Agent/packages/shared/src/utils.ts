import { VALIDATION } from './constants';

// Date utilities
export function formatDate(date: Date | string, format: string): string {
  const d = typeof date === 'string' ? new Date(date) : date;

  const tokens: Record<string, () => string> = {
    'YYYY': () => d.getFullYear().toString(),
    'MM': () => (d.getMonth() + 1).toString().padStart(2, '0'),
    'DD': () => d.getDate().toString().padStart(2, '0'),
    'HH': () => d.getHours().toString().padStart(2, '0'),
    'mm': () => d.getMinutes().toString().padStart(2, '0'),
    'ss': () => d.getSeconds().toString().padStart(2, '0'),
    'MMM': () => d.toLocaleString('default', { month: 'short' }),
    'MMMM': () => d.toLocaleString('default', { month: 'long' }),
    'ddd': () => d.toLocaleString('default', { weekday: 'short' }),
    'dddd': () => d.toLocaleString('default', { weekday: 'long' }),
  };

  return format.replace(/YYYY|MM|DD|HH|mm|ss|MMM|MMMM|ddd|dddd/g, match => tokens[match]());
}

export function isValidDate(date: unknown): boolean {
  if (!(date instanceof Date) && typeof date !== 'string') return false;
  const d = typeof date === 'string' ? new Date(date) : date;
  return d instanceof Date && !isNaN(d.getTime());
}

// Validation utilities
export function isValidEmail(email: string): boolean {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

export function isValidPassword(password: string): boolean {
  return password.length >= VALIDATION.MIN_PASSWORD_LENGTH &&
    /[A-Z]/.test(password) && // Has uppercase
    /[a-z]/.test(password) && // Has lowercase
    /[0-9]/.test(password) && // Has number
    /[^A-Za-z0-9]/.test(password); // Has special char
}

export function isValidLanguage(lang: string): boolean {
  return VALIDATION.SUPPORTED_LANGUAGES.includes(lang as any);
}

export function isValidTimezone(timezone: string): boolean {
  return VALIDATION.SUPPORTED_TIMEZONES.includes(timezone as any);
}

// String utilities
export function truncate(str: string, length: number): string {
  if (str.length <= length) return str;
  return str.slice(0, length) + '...';
}

export function slugify(str: string): string {
  return str
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/(^-|-$)/g, '');
}

// Object utilities
export function omit<T extends object, K extends keyof T>(
  obj: T,
  keys: K[]
): Omit<T, K> {
  const result = { ...obj };
  keys.forEach(key => delete result[key]);
  return result;
}

export function pick<T extends object, K extends keyof T>(
  obj: T,
  keys: K[]
): Pick<T, K> {
  const result = {} as Pick<T, K>;
  keys.forEach(key => {
    if (key in obj) result[key] = obj[key];
  });
  return result;
}

// Error utilities
export function isNetworkError(error: unknown): boolean {
  return error instanceof Error &&
    ('code' in error && error.code === 'NETWORK_ERROR' ||
      'message' in error && error.message.toLowerCase().includes('network'));
}

// Type guards
export function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function isString(value: unknown): value is string {
  return typeof value === 'string';
}

export function isNumber(value: unknown): value is number {
  return typeof value === 'number' && !isNaN(value);
}

// Async utilities
export async function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

export async function retry<T>(
  fn: () => Promise<T>,
  options: { attempts: number; delay: number }
): Promise<T> {
  let lastError: unknown;

  for (let i = 0; i < options.attempts; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (i < options.attempts - 1) {
        await sleep(options.delay);
      }
    }
  }

  throw lastError;
}

// Random ID generator
export function generateId(length = 16): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  return Array.from(
    { length },
    () => chars[Math.floor(Math.random() * chars.length)]
  ).join('');
}
