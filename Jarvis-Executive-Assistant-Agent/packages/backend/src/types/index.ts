import {
  User,
  UserPermission,
  UserPreferences,
  NotificationPreferences,
  UserIntegrations,
} from '@jarvis-executive-assistant/shared';

export {
  User,
  UserPermission,
  UserPreferences,
  NotificationPreferences,
  UserIntegrations,
};

// Skill related types
export interface Skill {
  name: string;
  description: string;
  parameters: SkillParameter[];
  execute: (params: Record<string, unknown>, context: SkillContext) => Promise<SkillResult>;
}

export interface SkillParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'object' | 'array';
  description: string;
  required: boolean;
}

export interface SkillContext {
  user: User;
  conversationId: string;
  messageId: string;
}

export interface SkillResult {
  success: boolean;
  data?: unknown;
  error?: string;
  message: string;
}

// LLM Agent types
export interface AgentMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: Record<string, unknown>;
}

export interface Conversation {
  id: string;
  userId: string;
  messages: AgentMessage[];
  context: ConversationContext;
  createdAt: Date;
  updatedAt: Date;
}

export interface ConversationContext {
  id: string;
  skills: string[];
  memory: Record<string, unknown>;
  activeTask?: TaskInfo;
}

export interface TaskInfo {
  id: string;
  type: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  data: Record<string, unknown>;
  createdAt: Date;
  updatedAt: Date;
}

// API Response types
export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: unknown;
  };
}

export interface PaginatedResponse<T> extends ApiResponse {
  data: {
    items: T[];
    total: number;
    page: number;
    pageSize: number;
  };
}

// Event types for WebSocket communication
export interface WebSocketEvent {
  type: string;
  payload: unknown;
  timestamp: Date;
}

export interface AgentEvent extends WebSocketEvent {
  type: 'thinking' | 'executing_skill' | 'completed' | 'error';
  payload: {
    conversationId: string;
    messageId: string;
    data: unknown;
  };
}

// Error types
export class AppError extends Error {
  constructor(
    public code: string,
    message: string,
    public statusCode: number = 500,
    public details?: unknown
  ) {
    super(message);
    this.name = 'AppError';
  }
}
