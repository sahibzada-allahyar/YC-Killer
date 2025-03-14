import OpenAI from 'openai';
import {
  LLMConfig,
  LLMResponse,
  ToolCall
} from '@ai-tutor/shared';
import { ITool } from '../tools/tool.interface';

export class LLMAgent {
  private openai: OpenAI;
  private tools: ITool[] = [];

  constructor(apiKey: string, private config: LLMConfig) {
    this.openai = new OpenAI({ apiKey });
  }

  public attachTool(tool: ITool): void {
    this.tools.push(tool);
  }

  public getAttachedTools(): string[] {
    return this.tools.map(tool => tool.name);
  }

  private buildToolDefinitions() {
    return this.tools.map(tool => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: {
          type: 'object',
          properties: tool.parameters.properties || {},
          required: tool.parameters.required || []
        }
      }
    }));
  }

  public async processQuery(input: string): Promise<LLMResponse> {
    try {
      const completion = await this.openai.chat.completions.create({
        model: this.config.model,
        messages: [{ role: 'user', content: input }],
        temperature: this.config.temperature,
        max_tokens: this.config.maxTokens,
        tools: this.buildToolDefinitions(),
        tool_choice: 'auto'
      });

      const toolCalls: ToolCall[] = [];

      // Process tool calls if any
      if (completion.choices[0]?.message.tool_calls) {
        for (const toolCall of completion.choices[0].message.tool_calls) {
          const tool = this.tools.find(t => t.name === toolCall.function.name);
          if (tool) {
            try {
              const result = await tool.execute(toolCall.function.arguments);
              toolCalls.push({
                toolName: tool.name,
                input: toolCall.function.arguments,
                result: JSON.stringify(result)
              });
            } catch (error) {
              toolCalls.push({
                toolName: tool.name,
                input: toolCall.function.arguments,
                error: error instanceof Error ? error.message : 'Tool execution failed'
              });
            }
          }
        }
      }

      return {
        content: completion.choices[0]?.message.content || '',
        toolCalls,
        usage: completion.usage ? {
          promptTokens: completion.usage.prompt_tokens,
          completionTokens: completion.usage.completion_tokens,
          totalTokens: completion.usage.total_tokens
        } : undefined
      };
    } catch (error) {
      throw new Error(
        `LLM processing error: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }
}
