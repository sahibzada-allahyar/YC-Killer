import FirecrawlApp, { SearchResponse } from '@mendable/firecrawl-js';
import pLimit from 'p-limit';
import { z } from 'zod';
import { generateObject } from 'ai';
import { knowledgeProcessor as o3MiniModel } from '../providers.js';
import { generateSystemPrompt } from '../prompt.js';
import { generateSearchQueries, processSearchResults } from '../utils/search.js';
import { InvestigationProgress, InvestigationOutcome } from '../types/research.js';
import { SearchData } from '../types/search.js';
import { OutputManager } from '../utils/outputmanagement.js';

// Initialize output manager for coordinated console/progress output
const outputManager = new OutputManager();

// Replace console.log with outputManager.log
function logMessage(...args: any[]) {
  outputManager.log(...args);
}

// Adjust this based on API rate limits
const ParallelExecutionLimit = 2;

// Initialize Firecrawl with optional API key and optional base url
const knowledgeEngine = new FirecrawlApp({
  apiKey: process.env.FIRECRAWL_KEY ?? '',
  apiUrl: process.env.FIRECRAWL_BASE_URL,
});

interface AnalysisReportResponse {
  reportMarkdown: string;
}

export async function generateAnalysisReport({
  initialQuery,
  insights,
  exploredSources,
}: {
  initialQuery: string;
  insights: string[];
  exploredSources: string[];
}): Promise<string> {
  const insightsBlock = insights
    .map(insight => `<insight>\n${insight}\n</insight>`)
    .join('\n');

  const response = await generateObject<AnalysisReportResponse>({
    model: o3MiniModel,
    system: generateSystemPrompt(),
    prompt: `Synthesize a comprehensive analysis report based on the following research prompt and gathered insights. Aim for extensive detail spanning 3+ pages, incorporating ALL research findings:\n\n<prompt>${initialQuery}</prompt>\n\nCollated Research Insights:\n\n<insights>\n${insightsBlock}\n</insights>`,
    schema: z.object({
      reportMarkdown: z
        .string()
        .describe('Comprehensive analysis report in Markdown format'),
    }),
  });

  // Append reference sources
  const sourcesSection = `\n\n## Reference Sources\n\n${exploredSources.map(source => `- ${source}`).join('\n')}`;
  return response.object.reportMarkdown + sourcesSection;
}

export async function conductInvestigation({
  query,
  expansionWidth,
  explorationDepth,
  insights = [],
  exploredSources = [],
  onProgress,
}: {
  query: string;
  expansionWidth: number;
  explorationDepth: number;
  insights?: string[];
  exploredSources?: string[];
  onProgress?: (progress: InvestigationProgress) => void;
}): Promise<InvestigationOutcome> {
  const progress: InvestigationProgress = {
    currentLevel: explorationDepth,
    totalLevels: explorationDepth,
    currentScope: expansionWidth,
    totalScope: expansionWidth,
    totalInquiries: 0,
    completedInquiries: 0,
  };
  
  const updateProgress = (update: Partial<InvestigationProgress>) => {
    Object.assign(progress, update);
    onProgress?.(progress);
  };

  const searchQueries = await generateSearchQueries({
    inquiry: query,
    queryCount: expansionWidth,
    previousInsights: insights,
  });
  
  updateProgress({
    totalInquiries: searchQueries.length,
    currentInquiry: searchQueries[0]?.inquiry
  });
  
  const executionLimiter = pLimit(ParallelExecutionLimit);

  const results = await Promise.all(
    searchQueries.map((searchQuery: { inquiry: string; investigationIntent: string }) =>
      executionLimiter(async () => {
        try {
          const searchResult = await knowledgeEngine.search(searchQuery.inquiry, {
            timeout: 15000,
            limit: 5,
            scrapeOptions: { formats: ['markdown'] },
          });

          const discoveredSources = (searchResult.data as SearchResponse['data'])
            .map(item => item.url)
            .filter((url): url is string => typeof url === 'string');
          
          const newExpansionWidth = Math.ceil(expansionWidth / 2);
          const newExplorationDepth = explorationDepth - 1;

          const processedResults = await processSearchResults({
            inquiry: searchQuery.inquiry,
            searchData: searchResult as SearchData,
            followUpLimit: newExpansionWidth,
          });
          
          const aggregatedInsights = [...insights, ...processedResults.insights];
          const aggregatedSources = [...exploredSources, ...discoveredSources];

          if (newExplorationDepth > 0) {
            logMessage(
              `Deepening investigation, width: ${newExpansionWidth}, depth: ${newExplorationDepth}`,
            );

            updateProgress({
              currentLevel: newExplorationDepth,
              currentScope: newExpansionWidth,
              completedInquiries: progress.completedInquiries + 1,
              currentInquiry: searchQuery.inquiry,
            });

            const nextInquiry = `
            Previous investigation goal: ${searchQuery.investigationIntent}
            Additional research directions: ${processedResults.followUpInquiries.map((q: string) => `\n${q}`).join('')}
          `.trim();

            return conductInvestigation({
              query: nextInquiry,
              expansionWidth: newExpansionWidth,
              explorationDepth: newExplorationDepth,
              insights: aggregatedInsights,
              exploredSources: aggregatedSources,
              onProgress,
            });
          } else {
            updateProgress({
              currentLevel: 0,
              completedInquiries: progress.completedInquiries + 1,
              currentInquiry: searchQuery.inquiry,
            });
            return {
              insights: aggregatedInsights,
              exploredSources: aggregatedSources,
            };
          }
        } catch (error: any) {
          if (error.message?.includes('Timeout')) {
            logMessage(
              `Timeout encountered for query: ${searchQuery.inquiry}: `,
              error,
            );
          } else {
            logMessage(`Error processing query: ${searchQuery.inquiry}: `, error);
          }
          return {
            insights: [],
            exploredSources: [],
          };
        }
      }),
    ),
  );

  return {
    insights: [...new Set(results.flatMap((r: { insights?: string[] }) => r.insights ?? []))],
    exploredSources: [...new Set(results.flatMap((r: { exploredSources?: string[] }) => r.exploredSources ?? []))],
  };
}
