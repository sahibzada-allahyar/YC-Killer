import { BaseSkill } from '../core/base-skill';
import { SkillContext, SkillParameter, SkillResult, AppError } from '../types';
import env from '../config/env';

interface SearchResult {
  title: string;
  link: string;
  snippet: string;
  position: number;
}

export class WebSearchSkill extends BaseSkill {
  public name = 'webSearch';
  public description = 'Search the web for information';
  public parameters: SkillParameter[] = [
    {
      name: 'query',
      type: 'string',
      description: 'The search query',
      required: true,
    },
    {
      name: 'numResults',
      type: 'number',
      description: 'Number of results to return (max 10)',
      required: false,
    },
  ];

  public async execute(
    params: Record<string, unknown>,
    context: SkillContext
  ): Promise<SkillResult> {
    try {
      this.validateParameters(params);

      if (!env.SERPAPI_API_KEY) {
        throw new AppError(
          'SERPAPI_KEY_MISSING',
          'SerpAPI key is required for web search'
        );
      }

      const query = params.query as string;
      const numResults = Math.min(params.numResults as number || 5, 10);

      const searchResults = await this.performSearch(query, numResults);

      return this.createSuccessResult(
        searchResults,
        `Found ${searchResults.length} results for "${query}"`
      );
    } catch (error) {
      if (error instanceof AppError) {
        return this.createErrorResult(error.message);
      }
      return this.createErrorResult(
        error instanceof Error ? error.message : 'Unknown error occurred'
      );
    }
  }

  private async performSearch(query: string, numResults: number): Promise<SearchResult[]> {
    const url = new URL('https://serpapi.com/search');
    url.searchParams.append('api_key', env.SERPAPI_API_KEY as string);
    url.searchParams.append('q', query);
    url.searchParams.append('engine', 'google');
    url.searchParams.append('num', numResults.toString());

    const response = await fetch(url.toString());
    
    if (!response.ok) {
      throw new AppError(
        'SEARCH_ERROR',
        `Search request failed: ${response.statusText}`
      );
    }

    const data = await response.json();

    if (!data.organic_results) {
      throw new AppError(
        'SEARCH_ERROR',
        'No results found'
      );
    }

    return data.organic_results.slice(0, numResults).map((result: any, index: number) => ({
      title: result.title,
      link: result.link,
      snippet: result.snippet,
      position: index + 1,
    }));
  }

  private sanitizeSearchResults(results: any[]): SearchResult[] {
    return results.map((result, index) => ({
      title: this.sanitizeString(result.title),
      link: this.sanitizeString(result.link),
      snippet: this.sanitizeString(result.snippet),
      position: index + 1,
    }));
  }

  private sanitizeString(str: string): string {
    if (typeof str !== 'string') return '';
    // Remove HTML tags and decode HTML entities
    return str
      .replace(/<[^>]*>/g, '')
      .replace(/&amp;/g, '&')
      .replace(/&lt;/g, '<')
      .replace(/&gt;/g, '>')
      .replace(/&quot;/g, '"')
      .replace(/&#039;/g, "'")
      .trim();
  }
}
