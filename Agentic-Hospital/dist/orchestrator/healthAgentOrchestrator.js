"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.HealthAgentOrchestrator = void 0;
const primaryCareAgent_1 = require("../agents/primaryCareAgent");
const fitnessCoachAgent_1 = require("../agents/fitnessCoachAgent");
const lifestyleCoachAgent_1 = require("../agents/lifestyleCoachAgent");
const diabetesSpecialistAgent_1 = require("../agents/diabetesSpecialistAgent");
const nutritionAgent_1 = require("../agents/nutritionAgent");
const mockInsurance_1 = require("../services/mockInsurance");
class HealthAgentOrchestrator {
    constructor() {
        this.agents = [
            new primaryCareAgent_1.PrimaryCareAgent(),
            new fitnessCoachAgent_1.FitnessCoachAgent(),
            new lifestyleCoachAgent_1.LifestyleCoachAgent(),
            new diabetesSpecialistAgent_1.DiabetesSpecialistAgent(),
            new nutritionAgent_1.NutritionAgent()
        ];
        this.insuranceService = new mockInsurance_1.MockInsuranceService();
    }
    async processQuery(query) {
        try {
            // Process query with all agents in parallel
            const agentPromises = this.agents.map(async (agent) => {
                try {
                    const enrichedQuery = this.enrichQueryWithContext(query, agent.role);
                    const response = await agent.processQuery(enrichedQuery);
                    // Add insurance coverage information
                    response.insuranceCoverage = this.insuranceService.checkCoverage(response.message + ' ' + response.recommendations.join(' '));
                    return {
                        agentName: agent.name,
                        response
                    };
                }
                catch (error) {
                    console.error(`Error from ${agent.name}:`, error);
                    return {
                        agentName: agent.name,
                        response: {
                            message: `Error processing query for ${agent.name}`,
                            recommendations: [],
                            priority: 'low',
                            insuranceCoverage: {
                                covered: false,
                                estimatedCoverage: 0,
                                notes: 'Unable to determine coverage due to processing error'
                            }
                        }
                    };
                }
            });
            const responses = await Promise.all(agentPromises);
            // Sort responses by priority
            responses.sort((a, b) => {
                const priorityMap = { high: 3, medium: 2, low: 1 };
                return priorityMap[b.response.priority] - priorityMap[a.response.priority];
            });
            return {
                query: query.query,
                responses,
                timestamp: new Date().toISOString()
            };
        }
        catch (error) {
            console.error('Orchestrator error:', error);
            throw new Error('Failed to process query across agents');
        }
    }
    enrichQueryWithContext(query, agentRole) {
        let enrichedQuery = query.query;
        if (query.patientContext) {
            const context = query.patientContext;
            enrichedQuery += `\n\nPatient Context:
        ${context.age ? `Age: ${context.age}` : ''}
        ${context.diabetesType ? `Diabetes Type: ${context.diabetesType}` : ''}
        ${context.lastBloodSugar ? `Last Blood Sugar Reading: ${context.lastBloodSugar} mg/dL` : ''}
        ${context.medications ? `Current Medications: ${context.medications.join(', ')}` : ''}
      `;
        }
        // Add role-specific context hints
        enrichedQuery += `\n\nPlease provide advice from your perspective as a ${agentRole}.`;
        return enrichedQuery;
    }
    getAgents() {
        return this.agents;
    }
}
exports.HealthAgentOrchestrator = HealthAgentOrchestrator;
