"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.LifestyleCoachAgent = void 0;
const baseAgent_1 = require("./baseAgent");
class LifestyleCoachAgent extends baseAgent_1.BaseAgent {
    constructor() {
        super('Lifestyle Coach Agent', 'Wellness and Lifestyle Specialist', `You are an AI lifestyle coach specializing in helping diabetic patients maintain a healthy lifestyle. 
      Your role is to:
      - Provide guidance on stress management and its impact on blood sugar
      - Recommend sleep hygiene practices for better diabetes management
      - Suggest work-life balance strategies considering medical needs
      - Offer practical tips for social situations and diabetes management
      - Help develop sustainable healthy habits
      - Address emotional and psychological aspects of living with diabetes
      
      Format your responses with clear recommendations prefixed with "-" for easy parsing.
      Focus on realistic, sustainable lifestyle changes that support diabetes management.
      Consider the patient's whole life context when making recommendations.`);
    }
    async processQuery(query) {
        const response = await this.getAnthropicResponse(query);
        return this.parseResponse(response);
    }
    parseResponse(response) {
        const baseResponse = super.parseResponse(response);
        // Add lifestyle-specific context
        const lifestyleContext = "Remember that lifestyle changes take time. Start small and build gradually for lasting success.";
        return {
            ...baseResponse,
            message: `${baseResponse.message}\n\n${lifestyleContext}`,
            recommendations: [
                ...baseResponse.recommendations,
                lifestyleContext
            ]
        };
    }
}
exports.LifestyleCoachAgent = LifestyleCoachAgent;
