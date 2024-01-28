import { HealthResponse } from '../types/agents';
import { BaseAgent } from './baseAgent';

export class FitnessCoachAgent extends BaseAgent {
  constructor() {
    super(
      'Fitness Coach Agent',
      'Exercise Specialist',
      `You are an AI fitness coach specializing in personalized exercise programs. 
      Your role is to:
      - Provide safe and effective exercise recommendations
      - Design appropriate workout plans based on individual needs and goals
      - Suggest modifications to exercises based on fitness level and limitations
      - Advise on proper form, technique, and progression
      - Recommend exercise intensity and duration based on client context
      
      Format your responses with clear recommendations prefixed with "-" for easy parsing.
      Always prioritize safety and proper form in your exercise advice.
      Include guidance on warming up, cooling down, and injury prevention.`
    );
  }

  async processQuery(query: string): Promise<HealthResponse> {
    const prompt = `
    As a fitness coach, provide exercise recommendations for:
    "${query}"
    
    Consider:
    - Current fitness level and limitations
    - Exercise intensity and progression
    - Proper form and technique
    - Safety precautions and injury prevention
    - Balance of cardio, strength, and flexibility
    
    Provide specific, actionable advice in a clear format.
    `;

    const response = await this.getAnthropicResponse(prompt);
    return this.parseResponse(response);
  }

  protected parseResponse(response: string): HealthResponse {
    const baseResponse = super.parseResponse(response);
    
    // Add general exercise safety disclaimer
    const safetyDisclaimer = "Always consult with your healthcare provider before starting a new exercise program. Start slowly and progress gradually to avoid injury.";
    return {
      ...baseResponse,
      message: `${baseResponse.message}\n\n${safetyDisclaimer}`,
      recommendations: [...baseResponse.recommendations, safetyDisclaimer]
    };
  }
}
