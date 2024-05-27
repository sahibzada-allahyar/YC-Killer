import { 
  HealthAgent, 
  PatientQuery, 
  MultiAgentResponse,
  AgentResponse,
  ComplexityLevel,
  TeamDiscussion
} from '../types/agents';
import { ModeratorAgent } from '../agents/moderatorAgent';
import { TeamCoordinatorAgent } from '../agents/teamCoordinatorAgent';
import { PrimaryCareAgent } from '../agents/primaryCareAgent';
import { FitnessCoachAgent } from '../agents/fitnessCoachAgent';
import { LifestyleCoachAgent } from '../agents/lifestyleCoachAgent';
import { DiabetesSpecialistAgent } from '../agents/diabetesSpecialistAgent';
import { NutritionAgent } from '../agents/nutritionAgent';
import { MockInsuranceService } from '../services/mockInsurance';

export class HealthAgentOrchestrator {
  private agents: HealthAgent[];
  private insuranceService: MockInsuranceService;
  private moderator: ModeratorAgent;
  private coordinator: TeamCoordinatorAgent;

  constructor() {
    this.agents = [
      new PrimaryCareAgent(),
      new FitnessCoachAgent(),
      new LifestyleCoachAgent(),
      new DiabetesSpecialistAgent(),
      new NutritionAgent()
    ];
    this.moderator = new ModeratorAgent();
    this.coordinator = new TeamCoordinatorAgent();
    this.insuranceService = new MockInsuranceService();
  }

  async processQuery(query: PatientQuery): Promise<MultiAgentResponse> {
    try {
      // First, assess complexity with moderator
      const complexityAssessment = await this.moderator.processQuery(query.query);
      const complexity = complexityAssessment.complexity || 'low';

      let responses: AgentResponse[] = [];
      let teamDiscussion: TeamDiscussion | undefined;

      if (complexity === 'low') {
        // For low complexity, select the most relevant agent based on query content
        const mostRelevantAgent = this.selectRelevantAgents(query.query, 1)[0];
        if (mostRelevantAgent) {
          const response = await this.processAgentQuery(mostRelevantAgent, query);
          responses = [response];
        }
      } else {
        // For moderate/high complexity, use relevant specialists
        const relevantAgents = complexity === 'high' 
          ? this.agents 
          : this.selectRelevantAgents(query.query, 2);

        // Process query with relevant agents in parallel
        const agentPromises = relevantAgents.map(agent => this.processAgentQuery(agent, query));
        responses = await Promise.all(agentPromises);

        // Facilitate team discussion for moderate/high complexity cases
        teamDiscussion = await this.coordinator.facilitateTeamDiscussion(
          query.query,
          responses.map(r => r.response)
        );
      }

      // Sort responses by priority
      responses.sort((a, b) => {
        const priorityMap = { high: 3, medium: 2, low: 1 };
        return priorityMap[b.response.priority] - priorityMap[a.response.priority];
      });

      return {
        query: query.query,
        responses,
        timestamp: new Date().toISOString(),
        complexity,
        teamDiscussion
      };
    } catch (error) {
      console.error('Orchestrator error:', error);
      throw new Error('Failed to process query across agents');
    }
  }

  private async processAgentQuery(agent: HealthAgent, query: PatientQuery): Promise<AgentResponse> {
    try {
      const enrichedQuery = this.enrichQueryWithContext(query, agent.role);
      const response = await agent.processQuery(enrichedQuery);
      
      // Add insurance coverage information
      response.insuranceCoverage = this.insuranceService.checkCoverage(
        response.message + ' ' + response.recommendations.join(' ')
      );

      return {
        agentName: agent.name,
        response
      };
    } catch (error) {
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
  }

  private selectRelevantAgents(query: string, count: number): HealthAgent[] {
    // Enhanced relevance check with keyword mapping and weights
    const keywordMap = {
      'exercise': {
        keywords: ['fitness', 'workout', 'training', 'gym', 'cardio', 'strength', 'exercise'],
        weight: 3
      },
      'nutrition': {
        keywords: ['diet', 'food', 'eating', 'meal', 'nutrients', 'nutrition'],
        weight: 3
      },
      'lifestyle': {
        keywords: ['habits', 'sleep', 'stress', 'routine', 'daily', 'balance', 'wellness'],
        weight: 2
      },
      'medical': {
        keywords: ['pain', 'symptoms', 'condition', 'disease', 'treatment', 'headache', 'dizzy', 
                  'medication', 'doctor', 'health', 'medical', 'diagnosis', 'fever', 'chronic'],
        weight: 5 // Higher weight for medical terms
      }
    };

    const relevanceScores = this.agents.map(agent => {
      let score = 0;
      const queryLower = query.toLowerCase();
      const roleWords = agent.role.toLowerCase().split(' ');
      
      // Special handling for Primary Care Agent - prioritize for medical queries
      if (agent instanceof PrimaryCareAgent && 
          Object.entries(keywordMap).some(([category, {keywords}]) => 
            category === 'medical' && keywords.some(k => queryLower.includes(k)))) {
        score += 10; // Significant boost for medical queries
      }

      // Check agent role against query
      if (roleWords.some(word => queryLower.includes(word))) {
        score += 5;
      }

      // Check related keywords with weights
      Object.entries(keywordMap).forEach(([category, {keywords, weight}]) => {
        if (roleWords.some(word => word.includes(category.toLowerCase()))) {
          keywords.forEach(keyword => {
            if (queryLower.includes(keyword)) {
              score += weight;
            }
          });
        }
      });

      return { agent, score };
    });

    // Sort by relevance score and take top count
    return relevanceScores
      .sort((a, b) => b.score - a.score)
      .slice(0, count)
      .map(item => item.agent);
  }

  private enrichQueryWithContext(query: PatientQuery, agentRole: string): string {
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

  getAgents(): HealthAgent[] {
    return this.agents;
  }
}
