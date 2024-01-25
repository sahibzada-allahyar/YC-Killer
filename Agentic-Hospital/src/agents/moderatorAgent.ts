import { BaseAgent } from './baseAgent';
import { HealthResponse, ComplexityLevel } from '../types/agents';

export class ModeratorAgent extends BaseAgent {
  constructor() {
    super(
      "Medical Moderator",
      "Complexity Assessment Specialist",
      `You are a medical complexity assessment specialist. Your role is to:
      1. Analyze medical queries to determine their complexity level
      2. Classify them as:
         - low: Simple, well-defined issues manageable by a single doctor
         - moderate: Issues requiring 2-3 specialist perspectives
         - high: Complex cases needing full team collaboration
      3. Provide clear reasoning for your assessment`
    );
  }

  async processQuery(query: string): Promise<HealthResponse> {
    const prompt = `
    Analyze this medical query and classify its complexity:
    "${query}"
    
    Respond in this format:
    Complexity: [low/moderate/high]
    Reasoning: [brief explanation]
    Message: [summary of key points]
    Recommendations: [bullet points if any]
    `;

    const response = await this.getAnthropicResponse(prompt);
    const parsed = this.parseComplexityResponse(response);
    return parsed;
  }

  private parseComplexityResponse(response: string): HealthResponse {
    const complexityMatch = response.match(/Complexity:\s*(low|moderate|high)/i);
    const reasoningMatch = response.match(/Reasoning:\s*([^\n]+)/);
    const messageMatch = response.match(/Message:\s*([^\n]+)/);
    
    const complexity = (complexityMatch?.[1].toLowerCase() || 'low') as ComplexityLevel;
    const reasoning = reasoningMatch?.[1] || '';
    const message = messageMatch?.[1] || response;

    const recommendations = response
      .split('\n')
      .filter(line => line.trim().startsWith('-'))
      .map(line => line.replace('-', '').trim());

    return {
      message,
      recommendations,
      priority: this.mapComplexityToPriority(complexity),
      complexity,
      reasoning
    };
  }

  private mapComplexityToPriority(complexity: ComplexityLevel): 'low' | 'medium' | 'high' {
    switch (complexity) {
      case 'high': return 'high';
      case 'moderate': return 'medium';
      case 'low': return 'low';
    }
  }
}
