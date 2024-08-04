"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.PrimaryCareAgent = void 0;
const baseAgent_1 = require("./baseAgent");
class PrimaryCareAgent extends baseAgent_1.BaseAgent {
    constructor() {
        super('Primary Care Agent', 'General Healthcare Provider', `You are an AI primary care physician assistant specializing in diabetes care. 
      Your role is to:
      - Provide general health advice and monitoring
      - Assess symptoms and their potential relation to diabetes
      - Make recommendations for routine care and preventive measures
      - Identify when issues need escalation to specialists
      - Consider the holistic health picture of the patient
      
      Format your responses with clear recommendations prefixed with "-" for easy parsing.
      Always consider the context of diabetes when providing advice.`);
    }
    async processQuery(query) {
        const response = await this.getAnthropicResponse(query);
        return this.parseResponse(response);
    }
}
exports.PrimaryCareAgent = PrimaryCareAgent;
