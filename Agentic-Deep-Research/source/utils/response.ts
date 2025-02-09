import { z } from 'zod';
import { generateObject } from 'ai';
import { o3MiniModel } from './ai/providers';
import { systemPrompt } from './prompt';

interface FeedbackResponse {
  analysis: string;
  actionableSteps: string[];
}

export async function processFeedback(feedback: string, context: string): Promise<string> {
  const response = await generateObject<FeedbackResponse>({
    model: o3MiniModel,
    system: systemPrompt(),
    prompt: `Analyze this feedback in the context of the research results and suggest improvements:\n\nContext:\n${context}\n\nFeedback:\n${feedback}`,
    schema: z.object({
      analysis: z.string().describe('Analysis of the feedback and suggested improvements'),
      actionableSteps: z
        .array(z.string())
        .describe('List of specific actions to implement the improvements'),
    }),
  });

  return `
Analysis:
${response.object.analysis}

Actionable Steps:
${response.object.actionableSteps.map((step: string, index: number) => `${index + 1}. ${step}`).join('\n')}
  `.trim();
}
