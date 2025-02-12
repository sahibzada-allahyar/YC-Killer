export interface Call {
  id: string;
  phoneNumber: string;
  type: 'sales' | 'support';
  status: 'ringing' | 'in-progress' | 'completed' | 'failed';
  startTime: Date;
  endTime?: Date;
  outcome?: string;
  recordingUrl?: string;
  transcriptSegments: TranscriptSegment[];
}

export interface TranscriptSegment {
  id: string;
  callId: string;
  speaker: 'assistant' | 'customer';
  text: string;
  timestamp: Date;
}

export interface CallFeedback {
  id: string;
  callId: string;
  reviewerId: string;
  rating: number;
  comments?: string;
  createdAt: Date;
}

export interface CallMetrics {
  totalCalls: number;
  successRate: number;
  averageDuration: number;
  callsByType: {
    sales: number;
    support: number;
  };
  callsByOutcome: {
    [key: string]: number;
  };
}

export interface APIResponse<T> {
  success: boolean;
  error?: string;
  data?: T;
}

export interface User {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'supervisor';
  createdAt: Date;
}

export interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface WebSocketMessage {
  type: 'callStarted' | 'callEnded' | 'transcriptUpdate' | 'statusUpdate';
  data: any;
}

// Training Types
export interface TrainingMetrics {
  totalReviews: number;
  averageRating: number;
  improvementRate: number;
  commonFeedback: {
    category: string;
    count: number;
  }[];
}

// Telephony Types
export interface CallConfig {
  maxDuration: number;
  recordingEnabled: boolean;
  transcriptionEnabled: boolean;
  language: string;
}

// LLM Types
export interface ConversationContext {
  callId: string;
  type: 'sales' | 'support';
  history: {
    role: 'assistant' | 'customer';
    content: string;
  }[];
  metadata?: Record<string, any>;
}

export interface LLMResponse {
  text: string;
  confidence: number;
  suggestedActions?: string[];
  metadata?: Record<string, any>;
}
