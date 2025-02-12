import { z } from 'zod';
import { generateObject } from 'ai';
import { compact } from 'lodash-es';
import { knowledgeProcessor as o3MiniModel, optimizePromptLength as trimPrompt } from '../providers.js';
import { generateSystemPrompt as systemPrompt } from '../prompt.js';
import { SearchQueryMetadata, ProcessedInsight } from '../types/research.js';
import { SearchData, SearchItem } from '../types/search.js';

interface QueryGenerationResponse {
  queries: Array<{
    inquiry: string;
    investigationIntent: string;
  }>;
}

interface ProcessedSearchResponse {
  insights: string[];
  followUpInquiries: string[];
}

export async function generateSearchQueries({
  inquiry,
  queryCount = 3,
  previousInsights,
}: {
  inquiry: string;
  queryCount?: number;
  previousInsights?: string[];
}): Promise<SearchQueryMetadata[]> {
  const response = await generateObject<QueryGenerationResponse>({
    model: o3MiniModel,
    system: systemPrompt(),
    prompt: `Given the following inquiry, generate strategic search queries to investigate the topic. Return up to ${queryCount} queries, optimizing for unique perspectives: <prompt>${inquiry}</prompt>\n\n${
      previousInsights
        ? `Consider these previous insights for more targeted queries: ${previousInsights.join(
            '\n',
          )}`
        : ''
    }`,
    schema: z.object({
      queries: z
        .array(
          z.object({
            inquiry: z.string().describe('The search inquiry'),
            investigationIntent: z
              .string()
              .describe(
                'Elaborate on the investigation purpose and potential research directions. Detail specific aspects to explore and potential connections to uncover.',
              ),
          }),
        )
        .describe(`Collection of search queries, maximum ${queryCount}`),
    }),
  });

  return response.object.queries.slice(0, queryCount);
}

export async function processSearchResults({
  inquiry,
  searchData,
  insightLimit = 3,
  followUpLimit = 3,
}: {
  inquiry: string;
  searchData: SearchData;
  insightLimit?: number;
  followUpLimit?: number;
}): Promise<ProcessedInsight> {
  const contentSegments = compact(searchData.data.map((item: SearchItem) => item.markdown)).map(
    (content: string) => trimPrompt(content, 25_000),
  );

  const response = await generateObject<ProcessedSearchResponse>({
    model: o3MiniModel,
    abortSignal: AbortSignal.timeout(60_000),
    system: systemPrompt(),
    prompt: `Analyze these search results for the inquiry <inquiry>${inquiry}</inquiry>. Extract key insights and generate follow-up questions. Focus on unique, information-dense findings. Include specific entities, metrics, and dates when available.\n\n<content>${contentSegments
      .map((segment: string) => `<segment>\n${segment}\n</segment>`)
      .join('\n')}</content>`,
    schema: z.object({
      insights: z
        .array(z.string())
        .describe(`Key insights extracted, maximum ${insightLimit}`),
      followUpInquiries: z
        .array(z.string())
        .describe(
          `Strategic follow-up questions for deeper investigation, maximum ${followUpLimit}`,
        ),
    }),
  });

  return {
    insights: response.object.insights,
    followUpInquiries: response.object.followUpInquiries,
  };
}
