import { HealthResponse } from '../types/agents';
import { BaseAgent } from './baseAgent';

export class LifestyleCoachAgent extends BaseAgent {
  constructor() {
    super(
      'Lifestyle Coach Agent',
      'Lifestyle and Wellness Coach',
      `You are an AI lifestyle coach specializing in holistic wellness.
      Your role is to:
      - Provide guidance on healthy lifestyle habits
      - Recommend stress management techniques
      - Suggest sleep improvement strategies
      - Offer work-life balance advice
      - Help establish sustainable daily routines
      
      Format your responses with clear recommendations prefixed with "-" for easy parsing.
      Focus on practical, achievable lifestyle changes.
      Emphasize the importance of gradual, sustainable improvements.`
    );
  }

  async processQuery(query: string): Promise<HealthResponse> {
    const prompt = `
    As a lifestyle and wellness coach, provide recommendations for:
    "${query}"
    
    Consider:
    - Daily habits and routines
    - Stress management
    - Sleep quality
    - Work-life balance
    - Overall wellness
    
    Provide specific, actionable advice in a clear format.
    Focus on sustainable lifestyle changes that can be maintained long-term.
    `;

    const response = await this.getAnthropicResponse(prompt);
    return this.parseResponse(response);
  }

  protected parseResponse(response: string): HealthResponse {
    const baseResponse = super.parseResponse(response);
    
    // Add lifestyle coaching disclaimer
    const disclaimer = "These are general lifestyle recommendations. For medical concerns, please consult with a healthcare provider.";
    return {
      ...baseResponse,
      message: `${baseResponse.message}\n\n${disclaimer}`,
      recommendations: [...baseResponse.recommendations, disclaimer]
    };
  }
}
