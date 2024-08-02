export interface Message {
  id: string;
  conversationId: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  metadata?: {
    skillExecuted?: string;
    skillResult?: unknown;
    error?: string;
  };
}

export interface Conversation {
  id: string;
  userId: string;
  title?: string;
  messages: Message[];
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

export interface CreateConversationRequest {
  title?: string;
  initialMessage?: string;
}

export interface SendMessageRequest {
  content: string;
  metadata?: Record<string, unknown>;
}

export interface AgentEvent {
  type: 'thinking' | 'executing_skill' | 'completed' | 'error';
  payload: {
    conversationId: string;
    messageId: string;
    data: unknown;
  };
  timestamp: Date;
}

export interface ConversationListResponse {
  conversations: Array<{
    id: string;
    title?: string;
    lastMessage?: {
      content: string;
      timestamp: Date;
    };
    createdAt: Date;
    updatedAt: Date;
  }>;
  total: number;
  page: number;
  pageSize: number;
}

export interface ConversationFilters {
  search?: string;
  startDate?: Date;
  endDate?: Date;
  page?: number;
  pageSize?: number;
}
