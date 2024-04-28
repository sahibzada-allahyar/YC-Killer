import { HealthResponse } from '../types/agents';
import { BaseAgent } from './baseAgent';

export class PrimaryCareAgent extends BaseAgent {
  constructor() {
    super(
      'Primary Care Agent',
      'Primary Care Physician',
      `You are an AI primary care physician providing general medical advice.
      Your role is to:
      - Assess medical symptoms and conditions
      - Provide general health recommendations
      - Identify when specialist consultation is needed
      - Offer preventive care guidance
      - Consider patient context and medical history
      
      Format your responses with clear recommendations prefixed with "-" for easy parsing.
      Always prioritize patient safety and appropriate medical care.
      Include appropriate warnings and referral recommendations when needed.`
    );
  }

  async processQuery(query: string): Promise<HealthResponse> {
    const prompt = `
    As a primary care physician, provide medical advice for:
    "${query}"
    
    Consider:
    - Severity of symptoms/condition
    - Need for immediate care or specialist referral
    - General health recommendations
    - Preventive care measures
    - Lifestyle modifications if relevant
    
    Provide specific, actionable advice in a clear format.
    If symptoms suggest a serious condition, emphasize the need for proper medical evaluation.
    `;

    const response = await this.getAnthropicResponse(prompt);
    return this.parseResponse(response);
  }

  protected parseResponse(response: string): HealthResponse {
    const baseResponse = super.parseResponse(response);
    
    // Add medical disclaimer
    const medicalDisclaimer = "This advice is for informational purposes only and does not constitute medical diagnosis or treatment. Always consult with a qualified healthcare provider for medical advice.";
    return {
      ...baseResponse,
      message: `${baseResponse.message}\n\n${medicalDisclaimer}`,
      recommendations: [...baseResponse.recommendations, medicalDisclaimer]
    };
  }
}
