import { HealthResponse } from '../types/agents';
import { BaseAgent } from './baseAgent';

export class NutritionAgent extends BaseAgent {
  constructor() {
    super(
      'Nutrition Agent',
      'Diabetes Nutrition Specialist',
      `You are an AI nutritionist specializing in diabetes dietary management. 
      Your role is to:
      - Provide personalized dietary recommendations for blood sugar control
      - Explain glycemic index and its impact on blood sugar
      - Suggest meal planning and timing strategies
      - Offer guidance on carbohydrate counting
      - Recommend healthy food substitutions
      - Address dietary challenges in social situations
      - Provide guidance on managing diet during exercise
      
      Format your responses with clear recommendations prefixed with "-" for easy parsing.
      Focus on practical, sustainable dietary advice that helps manage blood sugar levels.
      Include information about the timing of meals and snacks when relevant.`
    );
  }

  async processQuery(query: string): Promise<HealthResponse> {
    const response = await this.getAnthropicResponse(query);
    return this.parseResponse(response);
  }

  protected parseResponse(response: string): HealthResponse {
    const baseResponse = super.parseResponse(response);
    
    // Add nutrition-specific context
    const nutritionDisclaimer = 
      "Individual responses to foods may vary. Monitor your blood sugar to understand how different foods affect you personally. " +
      "Consider working with a registered dietitian for personalized meal planning.";
    
    // Extract any mentioned foods and their glycemic impact if present
    const foodMentions = this.extractFoodMentions(response);
    const nutritionContext = foodMentions.length > 0 
      ? "\n\nGlycemic Impact Notes:\n" + foodMentions.join("\n")
      : "";

    return {
      ...baseResponse,
      message: `${baseResponse.message}\n\n${nutritionDisclaimer}${nutritionContext}`,
      recommendations: [
        ...baseResponse.recommendations,
        nutritionDisclaimer,
        ...foodMentions
      ]
    };
  }

  private extractFoodMentions(response: string): string[] {
    const lowGIFoods = ['beans', 'lentils', 'nuts', 'whole grains', 'vegetables'];
    const highGIFoods = ['white bread', 'rice', 'potato', 'sugar', 'candy'];
    
    const mentions: string[] = [];
    
    [...lowGIFoods, ...highGIFoods].forEach(food => {
      if (response.toLowerCase().includes(food)) {
        const isLowGI = lowGIFoods.includes(food);
        mentions.push(
          `- ${food}: ${isLowGI ? 'Low' : 'High'} glycemic impact - ${
            isLowGI 
              ? 'generally good choice that helps maintain stable blood sugar'
              : 'may cause rapid blood sugar rise, consider portions carefully'
          }`
        );
      }
    });
    
    return mentions;
  }
}
