export interface ApiError {
  code: string;
  message: string;
  details?: unknown;
}

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: ApiError;
}

export interface PaginatedResponse<T> extends ApiResponse {
  data: {
    items: T[];
    total: number;
    page: number;
    pageSize: number;
  };
}

export interface PaginationParams {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
}

export interface DateRangeParams {
  startDate?: string;
  endDate?: string;
}

export interface SearchParams {
  query?: string;
  filters?: Record<string, unknown>;
}

// Common API request/response types
export interface HealthCheckResponse extends ApiResponse {
  data: {
    status: 'ok' | 'error';
    version: string;
    timestamp: string;
    services: Record<string, {
      status: 'ok' | 'error';
      message?: string;
    }>;
  };
}

export interface ErrorResponse extends ApiResponse {
  error: {
    code: string;
    message: string;
    details?: unknown;
  };
}

export interface SuccessResponse extends ApiResponse {
  data: {
    message: string;
  };
}

// WebSocket event types
export interface WebSocketEvent {
  type: string;
  payload: unknown;
  timestamp: Date;
}

export interface WebSocketErrorEvent extends WebSocketEvent {
  type: 'error';
  payload: {
    code: string;
    message: string;
    details?: unknown;
  };
}

export interface WebSocketAuthEvent extends WebSocketEvent {
  type: 'auth';
  payload: {
    token: string;
  };
}
