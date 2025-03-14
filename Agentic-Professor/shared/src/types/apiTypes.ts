import { ErrorCode } from '../index';

export interface ApiResponse<T = unknown> {
  success: boolean;
  data?: T;
  error?: ApiError;
}

export class ApiError extends Error {
  constructor(
    message: string,
    public readonly code: ErrorCode,
    public readonly statusCode: number,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export interface HomeworkSubmission {
  studentId?: string;
  subject?: string;
  imageData?: string;
  additionalNotes?: string;
}

export interface ProcessedHomework {
  extractedText: string;
  confidence: number;
}

export interface TutorResponse {
  htmlContent: string;
  plainText: string;
  audioUrl?: string;
  references?: string[];
  diagrams?: {
    id: string;
    type: 'image' | 'graph' | 'plot';
    data: string;
    caption?: string;
  }[];
}
