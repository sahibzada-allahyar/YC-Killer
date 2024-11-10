import OpenAI from 'openai';
import { config } from '../config';
import { CallType, LLMResponse, CallOutcome, TranscriptSegment } from '../utils/types';

class LLMService {
  private openai: OpenAI;
  
  constructor() {
    this.openai = new OpenAI({
      apiKey: config.openaiApiKey
    });
  }

  private async createChatCompletion(
    messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }>,
    temperature: number = 0.7
  ): Promise<{ text: string; confidence: number }> {
    try {
      const completion = await this.openai.chat.completions.create({
        model: 'gpt-4',
        messages,
        temperature,
        max_tokens: 150,
        top_p: 1,
        frequency_penalty: 0,
        presence_penalty: 0
      });

      // Calculate confidence based on response properties
      // Lower temperature and higher top_p indicate higher confidence
      // Also consider if it's a complete response (not truncated)
      const choice = completion.choices[0];
      const isComplete = choice?.finish_reason === 'stop';
      const confidenceScore = isComplete ? 
        Math.min(1, (1 - temperature) * 0.8 + 0.2) : // Higher confidence for lower temperature
        0.5; // Lower confidence for incomplete responses

      return {
        text: choice?.message?.content || '',
        confidence: confidenceScore
      };
    } catch (error) {
      console.error('Error calling OpenAI:', error);
      throw new Error('Failed to generate AI response');
    }
  }

  private getSystemPrompt(callType: CallType): string {
    if (callType === 'sales') {
      return `You are an AI sales agent for our company. Your goal is to:
1. Engage with the customer professionally and courteously
2. Understand their needs and pain points
3. Present our products/services that best match their needs
4. Handle objections effectively
5. Guide the conversation towards a sale or follow-up meeting
6. Close the sale when appropriate

Keep responses concise and natural. Ask questions to understand the customer's needs.
Be persuasive but not pushy. Focus on value proposition.`;
    } else {
      return `You are an AI customer support agent. Your goal is to:
1. Greet the customer professionally and empathetically
2. Understand their issue or concern clearly
3. Provide accurate and helpful solutions
4. Verify if the solution resolves their problem
5. Ensure customer satisfaction before ending the call

Keep responses clear and helpful. Ask clarifying questions when needed.
Show empathy and patience. Focus on resolving the issue efficiently.`;
    }
  }

  public async generateResponse(
    userMessage: string,
    callType: CallType,
    conversationHistory: TranscriptSegment[]
  ): Promise<LLMResponse> {
    // Construct conversation context
    const messages = [
      { role: 'system' as const, content: this.getSystemPrompt(callType) },
      // Add conversation history
      ...conversationHistory.map(segment => ({
        role: segment.speaker === 'user' ? 'user' as const : 'assistant' as const,
        content: segment.text
      })),
      // Add current user message
      { role: 'user' as const, content: userMessage }
    ];

    return this.createChatCompletion(messages);
  }

  public async classifyCallOutcome(transcript: TranscriptSegment[], callType: CallType): Promise<CallOutcome> {
    const transcriptText = transcript.map(t => `${t.speaker}: ${t.text}`).join('\n');
    const classificationPrompt = callType === 'sales' 
      ? 'Based on this sales call transcript, classify the outcome as either "SaleConfirmed" (customer agreed to purchase), "LeadPending" (customer showed interest but no immediate sale), or "NoSale" (customer declined). Respond with just the classification.'
      : 'Based on this support call transcript, classify the outcome as either "Resolved" (issue was fixed), "NeedsFollowUp" (requires additional action), or "Unresolved" (could not fix the issue). Respond with just the classification.';

    const messages = [
      { role: 'system' as const, content: 'You are a call outcome classifier. Analyze the transcript and provide the appropriate classification.' },
      { role: 'user' as const, content: `Transcript:\n${transcriptText}\n\n${classificationPrompt}` }
    ];

    const response = await this.createChatCompletion(messages, 0);
    
    // Validate and return the classification
    const validSalesOutcomes = ['SaleConfirmed', 'LeadPending', 'NoSale'];
    const validSupportOutcomes = ['Resolved', 'NeedsFollowUp', 'Unresolved'];
    const validOutcomes = callType === 'sales' ? validSalesOutcomes : validSupportOutcomes;
    
    const outcome = response.text.trim();
    if (!validOutcomes.includes(outcome as any)) {
      throw new Error(`Invalid classification: ${outcome}`);
    }

    return outcome as CallOutcome;
  }

  public async generateSummary(transcript: TranscriptSegment[]): Promise<string> {
    const transcriptText = transcript.map(t => `${t.speaker}: ${t.text}`).join('\n');
    const messages = [
      { role: 'system' as const, content: 'Create a brief, professional summary of this call. Focus on key points, decisions made, and next steps if any.' },
      { role: 'user' as const, content: transcriptText }
    ];

    const response = await this.createChatCompletion(messages, 0.3);
    return response.text;
  }
}

export const llmService = new LLMService();
