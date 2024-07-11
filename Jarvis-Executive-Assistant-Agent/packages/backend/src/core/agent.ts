import { OpenAI } from 'openai';
import { ChatCompletionMessageParam } from 'openai/resources/chat/completions';
import {
  AgentMessage,
  Skill,
  SkillResult,
  ConversationContext,
  AppError,
  User,
} from '../types';
import env from '../config/env';

export class LLMAgent {
  private openai: OpenAI;
  private skills: Map<string, Skill>;
  private context: ConversationContext;
  private user: User;

  constructor(user: User, context: ConversationContext) {
    this.openai = new OpenAI({ apiKey: env.OPENAI_API_KEY });
    this.skills = new Map();
    this.context = context;
    this.user = user;
  }

  public registerSkill(skill: Skill): void {
    this.skills.set(skill.name, skill);
  }

  public async processMessage(message: string): Promise<AgentMessage> {
    try {
      // Prepare conversation history for the LLM
      const messages = this.prepareMessages(message);

      // Get LLM response
      const completion = await this.openai.chat.completions.create({
        model: 'gpt-4',
        messages,
        temperature: 0.7,
        max_tokens: 1000,
      });

      const response = completion.choices[0]?.message?.content;
      if (!response) {
        throw new AppError('AGENT_ERROR', 'Failed to get response from LLM');
      }

      // Parse the response to check for skill execution
      const { shouldExecuteSkill, skillName, parameters } = this.parseResponse(response);

      let finalResponse = response;
      if (shouldExecuteSkill) {
        const skill = this.skills.get(skillName);
        if (!skill) {
          throw new AppError('SKILL_NOT_FOUND', `Skill ${skillName} not found`);
        }

        // Execute the skill
        const result = await this.executeSkill(skill, parameters);
        
        // Get a new response from LLM with the skill execution result
        finalResponse = await this.generateResponseWithResult(response, result);
      }

      // Create and return the agent's message
      const agentMessage: AgentMessage = {
        role: 'assistant',
        content: finalResponse,
        timestamp: new Date(),
        metadata: {
          skillExecuted: shouldExecuteSkill ? skillName : undefined,
          skillResult: shouldExecuteSkill ? parameters : undefined,
        },
      };

      return agentMessage;
    } catch (error) {
      if (error instanceof AppError) {
        throw error;
      }
      throw new AppError(
        'AGENT_ERROR',
        'Failed to process message',
        500,
        error instanceof Error ? error.message : 'Unknown error'
      );
    }
  }

  private prepareMessages(userMessage: string): ChatCompletionMessageParam[] {
    const systemPrompt = this.generateSystemPrompt();
    const messages: ChatCompletionMessageParam[] = [
      { role: 'system', content: systemPrompt },
    ];

    // Add relevant conversation history
    if (this.context.memory['recentMessages']) {
      const recentMessages = this.context.memory['recentMessages'] as AgentMessage[];
      messages.push(
        ...recentMessages.map(msg => ({
          role: msg.role,
          content: msg.content,
        }))
      );
    }

    // Add the current user message
    messages.push({ role: 'user', content: userMessage });

    return messages;
  }

  private generateSystemPrompt(): string {
    const availableSkills = Array.from(this.skills.values())
      .map(skill => `${skill.name}: ${skill.description}`)
      .join('\n');

    return `You are a helpful AI assistant with access to the following skills:

${availableSkills}

When you need to use a skill, format your response like this:
[SKILL_START]
{
  "name": "skillName",
  "parameters": {
    "param1": "value1",
    ...
  }
}
[SKILL_END]

Then continue your response normally. Always be clear, concise, and helpful.
User timezone: ${this.user.preferences?.timezone || 'UTC'}
User language: ${this.user.preferences?.language || 'en'}`;
  }

  private parseResponse(response: string): {
    shouldExecuteSkill: boolean;
    skillName: string;
    parameters: Record<string, unknown>;
  } {
    const skillMatch = response.match(/\[SKILL_START\]([\s\S]*?)\[SKILL_END\]/);
    
    if (!skillMatch) {
      return {
        shouldExecuteSkill: false,
        skillName: '',
        parameters: {},
      };
    }

    try {
      const skillData = JSON.parse(skillMatch[1]);
      return {
        shouldExecuteSkill: true,
        skillName: skillData.name,
        parameters: skillData.parameters,
      };
    } catch (error) {
      throw new AppError('PARSE_ERROR', 'Failed to parse skill execution data');
    }
  }

  private async executeSkill(
    skill: Skill,
    parameters: Record<string, unknown>
  ): Promise<SkillResult> {
    try {
      // Validate parameters against skill requirements
      this.validateParameters(skill, parameters);

      // Execute the skill with context
      return await skill.execute(parameters, {
        user: this.user,
        conversationId: this.context.id,
        messageId: Date.now().toString(), // You might want to generate this differently
      });
    } catch (error) {
      if (error instanceof AppError) {
        throw error;
      }
      throw new AppError(
        'SKILL_EXECUTION_ERROR',
        `Failed to execute skill ${skill.name}`,
        500,
        error instanceof Error ? error.message : 'Unknown error'
      );
    }
  }

  private validateParameters(
    skill: Skill,
    parameters: Record<string, unknown>
  ): void {
    for (const param of skill.parameters) {
      if (param.required && !(param.name in parameters)) {
        throw new AppError(
          'INVALID_PARAMETERS',
          `Missing required parameter: ${param.name}`
        );
      }
      // Add type validation if needed
    }
  }

  private async generateResponseWithResult(
    originalResponse: string,
    result: SkillResult
  ): Promise<string> {
    const completion = await this.openai.chat.completions.create({
      model: 'gpt-4',
      messages: [
        {
          role: 'system',
          content: 'Given the original response and skill execution result, generate a new response that incorporates the result naturally.',
        },
        {
          role: 'user',
          content: `Original response: ${originalResponse}\n\nSkill execution result: ${JSON.stringify(result)}`,
        },
      ],
      temperature: 0.7,
      max_tokens: 500,
    });

    return completion.choices[0]?.message?.content || originalResponse;
  }
}
