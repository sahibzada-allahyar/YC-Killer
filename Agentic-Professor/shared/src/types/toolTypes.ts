export interface ToolParameters {
  type: 'object';
  properties: Record<string, {
    type: 'string' | 'number' | 'boolean' | 'object';
    description: string;
    enum?: string[];
    minimum?: number;
    maximum?: number;
    pattern?: string;
  }>;
  required: string[];
  additionalProperties?: boolean;
}

export interface ToolResult {
  success: boolean;
  data: unknown;
  metadata?: {
    executionTime?: number;
    resourcesUsed?: string[];
    cacheHit?: boolean;
  };
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: {
    type: 'string' | 'number' | 'boolean' | 'object';
    description: string;
    required: boolean;
    schema?: Record<string, unknown>;
  }[];
}

export interface ToolConfig {
  maxRetries?: number;
  timeout?: number;
  rateLimit?: number;
  cacheEnabled?: boolean;
  cacheDuration?: number;
  defaultLanguage?: string;
}
