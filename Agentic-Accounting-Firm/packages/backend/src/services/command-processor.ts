import { 
  ParsedCommand,
  CommandContext,
  CommandResult,
  CommandStatus,
  CommandIntent
} from '@accounting-agent/shared';
import { openai } from './openai';
import { accountingService } from './accounting/accounting-service';
import { commandLogger as logger } from '../utils/logger';

class CommandProcessor {
  private static instance: CommandProcessor;

  private constructor() {}

  public static getInstance(): CommandProcessor {
    if (!CommandProcessor.instance) {
      CommandProcessor.instance = new CommandProcessor();
    }
    return CommandProcessor.instance;
  }

  public async processCommand(text: string, context: CommandContext): Promise<CommandResult> {
    const startTime = Date.now();
    let status: CommandStatus = 'processing';
    let result: CommandResult;

    try {
      logger.info('Processing command', { text, context });

      // Parse command using OpenAI
      const parsedCommand = await openai.parseCommand(text, context);
      
      // Execute command
      const data = await this.executeCommand(parsedCommand);

      result = {
        id: crypto.randomUUID(),
        status: 'completed',
        command: parsedCommand,
        result: {
          success: true,
          data,
        },
        executionTime: Date.now() - startTime,
        timestamp: new Date().toISOString(),
      };

    } catch (error) {
      logger.error('Failed to process command', { text, context, error });

      status = error instanceof Error && error.message.includes('clarification')
        ? 'requires_clarification'
        : 'failed';

      result = {
        id: crypto.randomUUID(),
        status,
        command: {
          originalText: text,
          intent: 'unknown' as CommandIntent,
          confidence: 0,
          entities: {},
          context,
        },
        result: {
          success: false,
          data: null,
          error: {
            code: 'COMMAND_PROCESSING_ERROR',
            message: error instanceof Error ? error.message : 'Failed to process command',
            details: error,
          },
        },
        executionTime: Date.now() - startTime,
        timestamp: new Date().toISOString(),
      };
    }

    logger.info('Command processing complete', { 
      commandId: result.id,
      status: result.status,
      executionTime: result.executionTime 
    });

    return result;
  }

  private async executeCommand(command: ParsedCommand): Promise<unknown> {
    logger.info('Executing command', { command });

    // Execute command based on intent
    return accountingService.executeCommand({
      intent: command.intent,
      entities: command.entities,
    });
  }

  public async transcribeAudio(audioBuffer: Buffer): Promise<string> {
    try {
      logger.info('Transcribing audio command');
      return await openai.transcribeAudio(audioBuffer);
    } catch (error) {
      logger.error('Failed to transcribe audio', { error });
      throw error;
    }
  }

  public async generateSuggestions(userId: string, context?: string): Promise<string[]> {
    try {
      return await openai.generateCommandSuggestions(userId, context);
    } catch (error) {
      logger.error('Failed to generate suggestions', { userId, context, error });
      return [];
    }
  }
}

export const commandProcessor = CommandProcessor.getInstance();
