import { HealthAgent, HealthResponse } from '../types/agents';
import { Anthropic } from '@anthropic-ai/sdk';
import dotenv from 'dotenv';

dotenv.config();

const anthropic = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY || '',
});

export abstract class BaseAgent implements HealthAgent {
  constructor(
    public name: string,
    public role: string,
    protected systemPrompt: string
  ) {}

  protected async getAnthropicResponse(query: string): Promise<string> {
    try {
      const response = await anthropic.beta.messages.create({
        model: 'claude-3-opus-20240229',
        max_tokens: 1000,
        messages: [
          {
            role: 'user',
            content: query
          }
        ],
        system: this.systemPrompt
      });

      const textContent = response.content.find(block => 'type' in block && block.type === 'text');
      if (!textContent || !('text' in textContent)) {
        throw new Error('No text content in response');
      }

      return textContent.text;
    } catch (error) {
      console.error('Error calling Anthropic API:', error);
      throw new Error('Failed to get response from AI model');
    }
  }

  abstract processQuery(query: string): Promise<HealthResponse>;

  protected parseResponse(response: string): HealthResponse {
    try {
      // Basic response parsing - in a real implementation, we'd want more sophisticated parsing
      const recommendations = response.split('\n').filter(line => line.trim().startsWith('-'));
      
      return {
        message: response,
        recommendations: recommendations.map(r => r.replace('-', '').trim()),
        priority: this.determinePriority(response)
      };
    } catch (error) {
      console.error('Error parsing agent response:', error);
      return {
        message: response,
        recommendations: [],
        priority: 'medium'
      };
    }
  }

  private determinePriority(response: string): 'low' | 'medium' | 'high' {
    const lowPriorityKeywords = ['maintain', 'continue', 'regular', 'routine'];
    const highPriorityKeywords = ['immediate', 'urgent', 'emergency', 'critical', 'severe'];

    const responseLower = response.toLowerCase();
    
    if (highPriorityKeywords.some(keyword => responseLower.includes(keyword))) {
      return 'high';
    }
    if (lowPriorityKeywords.some(keyword => responseLower.includes(keyword))) {
      return 'low';
    }
    return 'medium';
  }
}
