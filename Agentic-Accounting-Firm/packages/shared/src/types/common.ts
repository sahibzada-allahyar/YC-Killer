// Pagination types
export interface PaginationParams {
  page?: number;
  limit?: number;
  offset?: number;
}

export interface PaginatedResponse<T> {
  data: T[];
  pagination: {
    total: number;
    page: number;
    limit: number;
    hasMore: boolean;
  };
}

// Sort and filter types
export interface SortParams {
  field: string;
  order: 'asc' | 'desc';
}

export interface FilterOperator {
  eq?: unknown;
  ne?: unknown;
  gt?: number | Date;
  gte?: number | Date;
  lt?: number | Date;
  lte?: number | Date;
  in?: unknown[];
  nin?: unknown[];
  contains?: string;
  startsWith?: string;
  endsWith?: string;
}

export interface FilterParams {
  [field: string]: FilterOperator | unknown;
}

// API Response types
export interface ApiResponse<T> {
  success: boolean;
  data: T;
  error?: ApiError;
  meta?: Record<string, unknown>;
}

export interface ApiError {
  code: string;
  message: string;
  details?: unknown;
  stack?: string;
}

// Date range type
export interface DateRange {
  start: string;
  end: string;
}

// Money type
export interface Money {
  amount: number;
  currency: string;
}

// Address type
export interface Address {
  street1: string;
  street2?: string;
  city: string;
  state?: string;
  postalCode: string;
  country: string;
  type?: 'billing' | 'shipping' | 'business' | 'home';
}

// Contact information type
export interface ContactInfo {
  name: string;
  email: string;
  phone?: string;
  title?: string;
  department?: string;
}

// Audit fields
export interface AuditFields {
  createdAt: string;
  createdBy: string;
  updatedAt: string;
  updatedBy: string;
  version: number;
}

// Status tracking
export interface StatusTracking {
  current: string;
  previous?: string;
  history: Array<{
    status: string;
    timestamp: string;
    updatedBy: string;
    reason?: string;
  }>;
}

// File attachment
export interface FileAttachment {
  id: string;
  name: string;
  type: string;
  size: number;
  url: string;
  uploadedAt: string;
  uploadedBy: string;
}

// Search result
export interface SearchResult<T> {
  item: T;
  score: number;
  highlights?: Array<{
    field: string;
    snippet: string;
  }>;
}

// Webhook types
export interface WebhookPayload<T = unknown> {
  event: string;
  timestamp: string;
  data: T;
  metadata?: Record<string, unknown>;
}

export interface WebhookSubscription {
  id: string;
  url: string;
  events: string[];
  active: boolean;
  secret?: string;
  createdAt: string;
  updatedAt: string;
}

// Cache control
export interface CacheConfig {
  ttl: number;
  staleWhileRevalidate?: number;
  tags?: string[];
  group?: string;
}

// Rate limiting
export interface RateLimitConfig {
  points: number;
  duration: number;
  blockDuration?: number;
}

// Feature flags
export interface FeatureFlag {
  name: string;
  enabled: boolean;
  rules?: Array<{
    condition: Record<string, unknown>;
    value: boolean;
  }>;
  metadata?: Record<string, unknown>;
}

// Notification
export interface Notification {
  id: string;
  type: string;
  title: string;
  message: string;
  severity: 'info' | 'success' | 'warning' | 'error';
  read: boolean;
  createdAt: string;
  metadata?: Record<string, unknown>;
}

// User preferences
export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  language: string;
  timezone: string;
  dateFormat: string;
  numberFormat: string;
  notifications: {
    email: boolean;
    push: boolean;
    sms: boolean;
  };
}
