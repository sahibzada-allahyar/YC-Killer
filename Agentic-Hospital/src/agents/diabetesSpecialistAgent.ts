import { HealthResponse } from '../types/agents';
import { BaseAgent } from './baseAgent';

export class DiabetesSpecialistAgent extends BaseAgent {
  constructor() {
    super(
      'Diabetes Specialist Agent',
      'Endocrinologist & Diabetes Expert',
      `You are an AI diabetes specialist focusing on advanced diabetes management. 
      Your role is to:
      - Provide specialized advice on diabetes management
      - Interpret blood sugar patterns and suggest adjustments
      - Recommend medication timing and dosage considerations
      - Advise on the latest diabetes management technologies
      - Handle complex diabetes-related queries
      - Identify potential complications and warning signs
      - Guide on insulin management and adjustment strategies
      
      Format your responses with clear recommendations prefixed with "-" for easy parsing.
      Prioritize medical accuracy and safety in all recommendations.
      Always include relevant blood sugar monitoring advice.`
    );
  }

  async processQuery(query: string): Promise<HealthResponse> {
    const response = await this.getAnthropicResponse(query);
    return this.parseResponse(response);
  }

  protected parseResponse(response: string): HealthResponse {
    const baseResponse = super.parseResponse(response);
    
    // Add specialist-specific medical disclaimer
    const medicalDisclaimer = 
      "This advice is for informational purposes only. Always consult with your healthcare provider " +
      "before making changes to your diabetes management plan, especially regarding medication or insulin adjustments.";
    
    return {
      ...baseResponse,
      message: `${baseResponse.message}\n\n${medicalDisclaimer}`,
      recommendations: [
        ...baseResponse.recommendations,
        medicalDisclaimer
      ],
      // Specialist responses often need higher priority due to medical nature
      priority: this.adjustPriority(baseResponse.priority, response)
    };
  }

  private adjustPriority(
    basePriority: 'low' | 'medium' | 'high',
    response: string
  ): 'low' | 'medium' | 'high' {
    // Upgrade priority if certain critical keywords are present
    const criticalKeywords = [
      'ketoacidosis',
      'hypoglycemia',
      'hyperglycemia',
      'emergency',
      'immediate attention',
      'severe'
    ];

    if (criticalKeywords.some(keyword => 
      response.toLowerCase().includes(keyword))) {
      return 'high';
    }

    return basePriority;
  }
}
