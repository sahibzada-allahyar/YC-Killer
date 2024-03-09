"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.FitnessCoachAgent = void 0;
const baseAgent_1 = require("./baseAgent");
class FitnessCoachAgent extends baseAgent_1.BaseAgent {
    constructor() {
        super('Fitness Coach Agent', 'Exercise Specialist', `You are an AI fitness coach specializing in exercise programs for diabetic patients. 
      Your role is to:
      - Provide safe exercise recommendations considering blood sugar levels
      - Design appropriate workout plans that help manage diabetes
      - Suggest modifications to exercises based on patient limitations
      - Advise on pre and post-workout nutrition and glucose monitoring
      - Recommend exercise intensity and duration based on patient context
      
      Format your responses with clear recommendations prefixed with "-" for easy parsing.
      Always prioritize safety and blood sugar management in your exercise advice.
      Include warning signs to watch for during exercise.`);
    }
    async processQuery(query) {
        const response = await this.getAnthropicResponse(query);
        return this.parseResponse(response);
    }
    parseResponse(response) {
        const baseResponse = super.parseResponse(response);
        // Add exercise-specific safety disclaimer
        const safetyDisclaimer = "Always consult with your healthcare provider before starting a new exercise program. Monitor your blood sugar before, during, and after exercise.";
        return {
            ...baseResponse,
            message: `${baseResponse.message}\n\n${safetyDisclaimer}`,
            recommendations: [...baseResponse.recommendations, safetyDisclaimer]
        };
    }
}
exports.FitnessCoachAgent = FitnessCoachAgent;
