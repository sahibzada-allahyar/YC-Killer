import { BaseAgent } from './baseAgent';
import { HealthResponse, CollaborationMessage, TeamDiscussion } from '../types/agents';

export class TeamCoordinatorAgent extends BaseAgent {
  constructor() {
    super(
      "Team Coordinator",
      "Medical Team Coordinator",
      `You are a medical team coordinator. Your role is to:
      1. Facilitate discussion between medical specialists
      2. Synthesize different perspectives into a cohesive conclusion
      3. Ensure all relevant aspects of the case are considered`
    );
  }

  async processQuery(query: string): Promise<HealthResponse> {
    const prompt = `
    As a medical team coordinator, synthesize the key points and provide a conclusion for:
    "${query}"
    
    Provide your response in this format:
    Summary: [key points]
    Conclusion: [final assessment]
    Recommendations: [bullet points]
    `;

    const response = await this.getAnthropicResponse(prompt);
    return this.parseResponse(response);
  }

  async facilitateTeamDiscussion(
    query: string,
    responses: HealthResponse[]
  ): Promise<TeamDiscussion> {
    const discussionPrompt = `
    Review these specialist responses to the query: "${query}"

    Specialist Responses:
    ${responses.map(r => `- ${r.message}\n${r.recommendations.map(rec => `  * ${rec}`).join('\n')}`).join('\n\n')}

    Synthesize a team conclusion that:
    1. Identifies key areas of agreement
    2. Resolves any contradictions
    3. Provides clear, actionable recommendations

    Respond in this format:
    Discussion Summary: [synthesis of key points]
    Team Conclusion: [final consensus]
    Action Items: [bullet points]
    `;

    const response = await this.getAnthropicResponse(discussionPrompt);
    
    const messages: CollaborationMessage[] = [{
      from: this.name,
      content: response,
      timestamp: new Date().toISOString()
    }];

    const conclusionMatch = response.match(/Team Conclusion:\s*([^\n]+)/);
    const conclusion = conclusionMatch?.[1] || response;

    return {
      messages,
      conclusion
    };
  }
}
