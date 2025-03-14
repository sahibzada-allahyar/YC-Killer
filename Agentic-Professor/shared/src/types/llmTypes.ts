export interface LLMConfig {
  model: string;
  temperature?: number;
  maxTokens?: number;
  topP?: number;
  frequencyPenalty?: number;
  presencePenalty?: number;
}

export interface LLMResponse {
  content: string;
  toolCalls?: ToolCall[];
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
}

export interface ToolCall {
  toolName: string;
  input: string;
  result?: string;
  error?: string;
}

export interface TutorState {
  activeTools: string[];
  processingState: 'idle' | 'processing' | 'speaking' | 'error';
  currentSession?: string;
  lastResponse?: LLMResponse;
  error?: string;
}

export interface Config {
  llm: LLMConfig;
  tools: ToolDefinition[];
  enabledFeatures: {
    voice: boolean;
    wikipedia: boolean;
    imageProcessing: boolean;
  };
}

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    description: string;
    required: boolean;
    schema?: Record<string, unknown>;
  }[];
}
