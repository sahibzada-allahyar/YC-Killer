import type {
  LLMConfig,
  LLMResponse,
  ToolCall,
  TutorState,
  Config as BaseConfig,
  ToolDefinition
} from './types/llmTypes';

export type { LLMConfig, LLMResponse, ToolCall, TutorState, ToolDefinition };

export {
  ApiError,
  type ApiResponse,
  type HomeworkSubmission,
  type ProcessedHomework,
  type TutorResponse
} from './types/apiTypes';

// Voice types
export interface VoiceConfig {
  model: string;
  voice: OpenAIVoice;
  speed?: number;
  pitch?: number;
}

export type OpenAIVoice = 'alloy' | 'echo' | 'fable' | 'onyx' | 'nova' | 'shimmer';

// Default configurations
export const DEFAULT_VOICE_CONFIG: VoiceConfig = {
  model: 'tts-1',
  voice: 'alloy',
  speed: 1.0,
  pitch: 1.0
};

// Error types
export enum ErrorCode {
  INVALID_REQUEST = 'INVALID_REQUEST',
  INTERNAL_ERROR = 'INTERNAL_ERROR',
  UNAUTHORIZED = 'UNAUTHORIZED',
  NOT_FOUND = 'NOT_FOUND',
  RATE_LIMITED = 'RATE_LIMITED'
}

export enum ToolErrorCode {
  INITIALIZATION_FAILED = 'INITIALIZATION_FAILED',
  EXECUTION_FAILED = 'EXECUTION_FAILED',
  INVALID_INPUT = 'INVALID_INPUT',
  RATE_LIMITED = 'RATE_LIMITED'
}

export class ToolError extends Error {
  constructor(
    message: string,
    public readonly toolName: string,
    public readonly code: ToolErrorCode,
    public readonly details?: Record<string, unknown>
  ) {
    super(message);
    this.name = 'ToolError';
  }
}

// Configuration types
export interface Config extends BaseConfig {
  voice: VoiceConfig;
}

// Default configuration
export const DEFAULT_CONFIG: Config = {
  llm: {
    model: 'gpt-4-turbo-preview',
    temperature: 0.7,
    maxTokens: 2048,
    topP: 1,
    frequencyPenalty: 0,
    presencePenalty: 0
  },
  tools: [],
  enabledFeatures: {
    voice: true,
    wikipedia: true,
    imageProcessing: true
  },
  voice: DEFAULT_VOICE_CONFIG
};

// Configuration utilities
export function mergeConfig(base: Config, override: Partial<Config>): Config {
  return {
    ...base,
    llm: {
      ...base.llm,
      ...override.llm
    },
    tools: [...base.tools, ...(override.tools || [])],
    enabledFeatures: {
      ...base.enabledFeatures,
      ...override.enabledFeatures
    },
    voice: override.voice
      ? { ...DEFAULT_VOICE_CONFIG, ...override.voice }
      : DEFAULT_VOICE_CONFIG
  };
}
