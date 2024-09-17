export interface SkillParameter {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'object' | 'array';
  description: string;
  required: boolean;
  defaultValue?: unknown;
}

export interface SkillContext {
  user: {
    id: string;
    email: string;
    preferences?: {
      timezone: string;
      language: string;
    };
  };
  conversationId: string;
  messageId: string;
}

export interface SkillResult {
  success: boolean;
  data?: unknown;
  error?: string;
  message: string;
}

export interface Skill {
  name: string;
  description: string;
  parameters: SkillParameter[];
  execute: (params: Record<string, unknown>, context: SkillContext) => Promise<SkillResult>;
}

// Calendar Skill Types
export interface CalendarEvent {
  id: string;
  title: string;
  description?: string;
  startTime: string;
  endTime: string;
  attendees?: string[];
  location?: string;
  status: 'confirmed' | 'tentative' | 'cancelled';
}

export interface CreateCalendarEventParams {
  title: string;
  startTime: string;
  endTime: string;
  description?: string;
  attendees?: string[];
  location?: string;
}

// Web Search Skill Types
export interface SearchResult {
  title: string;
  link: string;
  snippet: string;
  position: number;
}

export interface WebSearchParams {
  query: string;
  numResults?: number;
  language?: string;
}

// Email Skill Types
export interface EmailParams {
  to: string[];
  subject: string;
  body: string;
  cc?: string[];
  bcc?: string[];
  attachments?: Array<{
    filename: string;
    content: string;
    contentType: string;
  }>;
}

// Reminder Skill Types
export interface ReminderParams {
  title: string;
  dueDate: string;
  description?: string;
  priority?: 'low' | 'medium' | 'high';
  repeat?: 'daily' | 'weekly' | 'monthly' | 'yearly' | null;
}
