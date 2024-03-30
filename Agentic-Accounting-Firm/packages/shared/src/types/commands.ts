export interface CommandContext {
  userId: string;
  accountId?: string;
  transactionId?: string;
  metadata?: Record<string, unknown>;
}

export type CommandIntent =
  | 'create_transaction'
  | 'get_transactions'
  | 'generate_report'
  | 'get_account_balance'
  | 'categorize_transaction'
  | 'unknown';

export interface ParsedCommand {
  originalText: string;
  intent: CommandIntent;
  confidence: number;
  entities: Record<string, unknown>;
  context: CommandContext;
}

export interface CommandError {
  code: string;
  message: string;
  details?: unknown;
}

export type CommandStatus = 
  | 'processing'
  | 'completed'
  | 'failed'
  | 'requires_clarification';

export interface CommandResult {
  id: string;
  status: CommandStatus;
  command: ParsedCommand;
  result: {
    success: boolean;
    data: unknown | null;
    error?: CommandError;
  };
  executionTime: number;
  timestamp: string;
}

export interface CommandSuggestion {
  text: string;
  description: string;
  context?: Record<string, unknown>;
}

export interface CommandEntity {
  type: string;
  value: unknown;
  confidence: number;
  metadata?: Record<string, unknown>;
}

export interface CommandMetadata {
  source: 'text' | 'voice';
  locale?: string;
  timezone?: string;
  deviceInfo?: {
    type: string;
    os?: string;
    browser?: string;
  };
  userPreferences?: {
    dateFormat?: string;
    currency?: string;
    language?: string;
  };
}

export interface CommandValidationResult {
  isValid: boolean;
  errors?: Array<{
    field: string;
    message: string;
  }>;
  warnings?: Array<{
    field: string;
    message: string;
  }>;
}

export interface CommandExecutionOptions {
  dryRun?: boolean;
  timeout?: number;
  retryAttempts?: number;
  validateOnly?: boolean;
}

export interface CommandHistory {
  id: string;
  userId: string;
  command: ParsedCommand;
  result: CommandResult;
  timestamp: string;
  metadata?: CommandMetadata;
}

export interface CommandStats {
  totalCommands: number;
  successfulCommands: number;
  failedCommands: number;
  averageExecutionTime: number;
  commandsByIntent: Record<CommandIntent, number>;
  commandsByStatus: Record<CommandStatus, number>;
  lastExecutedAt?: string;
}

export interface CommandProcessor {
  processCommand(text: string, context: CommandContext): Promise<CommandResult>;
  transcribeAudio(audioBuffer: Buffer): Promise<string>;
  generateSuggestions(userId: string, context?: string): Promise<string[]>;
  validateCommand(command: ParsedCommand): CommandValidationResult;
  getCommandHistory(userId: string, limit?: number): Promise<CommandHistory[]>;
  getCommandStats(userId: string): Promise<CommandStats>;
}
