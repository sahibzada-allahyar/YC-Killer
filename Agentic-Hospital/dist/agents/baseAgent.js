"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.BaseAgent = void 0;
const sdk_1 = require("@anthropic-ai/sdk");
const dotenv_1 = __importDefault(require("dotenv"));
dotenv_1.default.config();
const anthropic = new sdk_1.Anthropic({
    apiKey: process.env.ANTHROPIC_API_KEY || '',
});
class BaseAgent {
    constructor(name, role, systemPrompt) {
        this.name = name;
        this.role = role;
        this.systemPrompt = systemPrompt;
    }
    async getAnthropicResponse(query) {
        try {
            const response = await anthropic.beta.messages.create({
                model: 'claude-3-opus-20240229',
                max_tokens: 1000,
                messages: [
                    {
                        role: 'user',
                        content: query
                    }
                ],
                system: this.systemPrompt
            });
            const textContent = response.content.find(block => 'type' in block && block.type === 'text');
            if (!textContent || !('text' in textContent)) {
                throw new Error('No text content in response');
            }
            return textContent.text;
        }
        catch (error) {
            console.error('Error calling Anthropic API:', error);
            throw new Error('Failed to get response from AI model');
        }
    }
    parseResponse(response) {
        try {
            // Basic response parsing - in a real implementation, we'd want more sophisticated parsing
            const recommendations = response.split('\n').filter(line => line.trim().startsWith('-'));
            return {
                message: response,
                recommendations: recommendations.map(r => r.replace('-', '').trim()),
                priority: this.determinePriority(response)
            };
        }
        catch (error) {
            console.error('Error parsing agent response:', error);
            return {
                message: response,
                recommendations: [],
                priority: 'medium'
            };
        }
    }
    determinePriority(response) {
        const lowPriorityKeywords = ['maintain', 'continue', 'regular', 'routine'];
        const highPriorityKeywords = ['immediate', 'urgent', 'emergency', 'critical', 'severe'];
        const responseLower = response.toLowerCase();
        if (highPriorityKeywords.some(keyword => responseLower.includes(keyword))) {
            return 'high';
        }
        if (lowPriorityKeywords.some(keyword => responseLower.includes(keyword))) {
            return 'low';
        }
        return 'medium';
    }
}
exports.BaseAgent = BaseAgent;
