// Call status types
export type CallStatus = 'ringing' | 'in-progress' | 'completed' | 'failed';

// Call type (inbound/outbound)
export type CallType = 'sales' | 'support';

// Call outcome classifications
export type SalesOutcome = 'SaleConfirmed' | 'LeadPending' | 'NoSale';
export type SupportOutcome = 'Resolved' | 'NeedsFollowUp' | 'Unresolved';
export type CallOutcome = SalesOutcome | SupportOutcome;

// Transcript segment representing a single utterance in the conversation
export interface TranscriptSegment {
  id: string;
  callId: string;
  timestamp: Date;
  speaker: 'user' | 'assistant';
  text: string;
}

// Complete call record
export interface Call {
  id: string;
  type: CallType;
  status: CallStatus;
  phoneNumber: string;
  startTime: Date;
  endTime?: Date;
  outcome?: CallOutcome;
  recordingUrl?: string;
  transcriptSegments: TranscriptSegment[];
}

// Feedback from human reviewers
export interface CallFeedback {
  id: string;
  callId: string;
  reviewerId: string;
  rating: number; // 1-5 scale
  comments?: string;
  createdAt: Date;
}

// Active call state maintained during conversation
export interface ActiveCallState {
  callId: string;
  type: CallType;
  transcript: TranscriptSegment[];
  lastUserMessage?: string;
  lastAssistantMessage?: string;
  conversationContext: string[];
}

// LLM service response format
export interface LLMResponse {
  text: string;
  confidence?: number;
  suggestedOutcome?: CallOutcome;
}

// WebSocket event types for real-time updates
export type WebSocketEventType = 
  | 'callStarted'
  | 'transcriptUpdate'
  | 'callEnded'
  | 'feedbackSubmitted';

export interface WebSocketEvent {
  type: WebSocketEventType;
  callId: string;
  data: any;
  timestamp: Date;
}

// Admin user type
export interface AdminUser {
  id: string;
  email: string;
  name: string;
  role: 'admin' | 'supervisor';
  passwordHash: string;
  createdAt: Date;
  updatedAt: Date;
}

// API response wrapper
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

// Pagination parameters
export interface PaginationParams {
  page: number;
  limit: number;
}

// Call filters for listing/searching
export interface CallFilters {
  type?: CallType;
  status?: CallStatus;
  outcome?: CallOutcome;
  startDate?: Date;
  endDate?: Date;
  phoneNumber?: string;
}

// Analytics data types
export interface CallMetrics {
  totalCalls: number;
  averageDuration: number;
  successRate: number;
  callsByType: Record<CallType, number>;
  callsByOutcome: Record<CallOutcome, number>;
}
