import fs from 'fs';
import OpenAI from 'openai';
import { CommandContext, CommandIntent, ParsedCommand } from '@accounting-agent/shared';
import { config } from '../config';
import { openaiLogger as logger } from '../utils/logger';
import { createTempFile } from '../utils/file';

class OpenAIService {
  private client: OpenAI;
  private static instance: OpenAIService;

  private constructor() {
    this.client = new OpenAI({
      apiKey: config.openai.apiKey,
      organization: config.openai.organization,
    });
  }

  public static getInstance(): OpenAIService {
    if (!OpenAIService.instance) {
      OpenAIService.instance = new OpenAIService();
    }
    return OpenAIService.instance;
  }

  public async parseCommand(text: string, context: CommandContext): Promise<ParsedCommand> {
    try {
      logger.info('Parsing command with OpenAI', { text });

      const prompt = this.buildCommandParsingPrompt(text, context);
      const response = await this.client.chat.completions.create({
        model: config.openai.model,
        temperature: config.openai.temperature,
        max_tokens: config.openai.maxTokens,
        messages: [
          {
            role: 'system',
            content: 'You are an AI assistant that helps parse natural language commands into structured accounting operations.',
          },
          {
            role: 'user',
            content: prompt,
          },
        ],
      });

      const result = response.choices[0]?.message?.content;
      if (!result) {
        throw new Error('Failed to get response from OpenAI');
      }

      const parsed = JSON.parse(result) as {
        intent: CommandIntent;
        confidence: number;
        entities: Record<string, any>;
      };

      return {
        originalText: text,
        intent: parsed.intent,
        confidence: parsed.confidence,
        entities: parsed.entities,
        context,
      };
    } catch (error) {
      logger.error('Failed to parse command with OpenAI', { error, text });
      throw error;
    }
  }

  public async transcribeAudio(audioBuffer: Buffer): Promise<string> {
    try {
      logger.info('Transcribing audio with OpenAI Whisper');

      // Create a temporary file for the audio
      const { path: tempPath, cleanup } = await createTempFile(
        audioBuffer,
        'audio.wav',
        'audio/wav'
      );

      try {
        const response = await this.client.audio.transcriptions.create({
          file: fs.createReadStream(tempPath),
          model: 'whisper-1',
          language: 'en',
        });

        return response.text;
      } finally {
        // Ensure we clean up the temporary file
        await cleanup();
      }
    } catch (error) {
      logger.error('Failed to transcribe audio with OpenAI', { error });
      throw error;
    }
  }

  public async generateCommandSuggestions(
    userId: string,
    context?: string
  ): Promise<string[]> {
    try {
      logger.info('Generating command suggestions', { userId, context });

      const prompt = this.buildSuggestionPrompt(userId, context);
      const response = await this.client.chat.completions.create({
        model: config.openai.model,
        temperature: 0.7,
        max_tokens: 200,
        messages: [
          {
            role: 'system',
            content: 'You are an AI assistant that suggests helpful accounting commands based on user context.',
          },
          {
            role: 'user',
            content: prompt,
          },
        ],
      });

      const result = response.choices[0]?.message?.content;
      if (!result) {
        throw new Error('Failed to get suggestions from OpenAI');
      }

      return JSON.parse(result) as string[];
    } catch (error) {
      logger.error('Failed to generate suggestions', { error, userId });
      throw error;
    }
  }

  private buildCommandParsingPrompt(text: string, context: CommandContext): string {
    return `
Parse the following natural language command into a structured format for accounting operations.
The response should be a JSON object with the following structure:
{
  "intent": string (one of: create_transaction, get_transactions, generate_report, get_account_balance, categorize_transaction, unknown),
  "confidence": number (0-1),
  "entities": object (extracted parameters like amount, date, category, etc.)
}

Command: ${text}

User Context:
${JSON.stringify(context, null, 2)}

Parse the command considering the user's context and return only the JSON response.
    `.trim();
  }

  private buildSuggestionPrompt(userId: string, context?: string): string {
    return `
Generate a list of 5 helpful accounting command suggestions for the user.
The suggestions should be natural language commands that the user can use.

User ID: ${userId}
${context ? `Context: ${context}` : ''}

Return the suggestions as a JSON array of strings.
    `.trim();
  }
}

export const openai = OpenAIService.getInstance();
