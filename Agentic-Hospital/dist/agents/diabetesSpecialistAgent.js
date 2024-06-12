"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.DiabetesSpecialistAgent = void 0;
const baseAgent_1 = require("./baseAgent");
class DiabetesSpecialistAgent extends baseAgent_1.BaseAgent {
    constructor() {
        super('Diabetes Specialist Agent', 'Endocrinologist & Diabetes Expert', `You are an AI diabetes specialist focusing on advanced diabetes management. 
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
      Always include relevant blood sugar monitoring advice.`);
    }
    async processQuery(query) {
        const response = await this.getAnthropicResponse(query);
        return this.parseResponse(response);
    }
    parseResponse(response) {
        const baseResponse = super.parseResponse(response);
        // Add specialist-specific medical disclaimer
        const medicalDisclaimer = "This advice is for informational purposes only. Always consult with your healthcare provider " +
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
    adjustPriority(basePriority, response) {
        // Upgrade priority if certain critical keywords are present
        const criticalKeywords = [
            'ketoacidosis',
            'hypoglycemia',
            'hyperglycemia',
            'emergency',
            'immediate attention',
            'severe'
        ];
        if (criticalKeywords.some(keyword => response.toLowerCase().includes(keyword))) {
            return 'high';
        }
        return basePriority;
    }
}
exports.DiabetesSpecialistAgent = DiabetesSpecialistAgent;
