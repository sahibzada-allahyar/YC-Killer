import axios from 'axios';
import { ITool } from '../tool.interface';

interface WikipediaSearchParams {
  query: string;
  limit?: number;
}

interface WikipediaResponse {
  title: string;
  extract: string;
  url: string;
  thumbnail?: {
    source: string;
    width: number;
    height: number;
  };
}

export class WikipediaTool implements ITool {
  public name = 'wikipedia';
  public description = 'Search Wikipedia for articles and information';
  public parameters = {
    properties: {
      query: {
        type: 'string',
        description: 'The search query'
      },
      limit: {
        type: 'number',
        description: 'Maximum number of results (1-10)',
        minimum: 1,
        maximum: 10,
        default: 1
      }
    },
    required: ['query']
  };

  private async searchWikipedia(query: string, limit: number = 1): Promise<WikipediaResponse[]> {
    try {
      // First search for articles
      const searchResponse = await axios.get('https://en.wikipedia.org/w/api.php', {
        params: {
          action: 'query',
          list: 'search',
          srsearch: query,
          format: 'json',
          origin: '*',
          srlimit: limit
        }
      });

      const pageIds = searchResponse.data.query.search
        .slice(0, limit)
        .map((result: any) => result.pageid);

      // Then get detailed information for each page
      const detailsResponse = await axios.get('https://en.wikipedia.org/w/api.php', {
        params: {
          action: 'query',
          pageids: pageIds.join('|'),
          prop: 'extracts|info|pageimages',
          exintro: true,
          explaintext: true,
          inprop: 'url',
          pithumbsize: 300,
          format: 'json',
          origin: '*'
        }
      });

      return Object.values(detailsResponse.data.query.pages).map((page: any) => ({
        title: page.title,
        extract: page.extract,
        url: page.fullurl,
        thumbnail: page.thumbnail ? {
          source: page.thumbnail.source,
          width: page.thumbnail.width,
          height: page.thumbnail.height
        } : undefined
      }));
    } catch (error) {
      throw new Error(
        `Wikipedia API error: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  public async execute(input: string): Promise<WikipediaResponse[]> {
    try {
      const params: WikipediaSearchParams = JSON.parse(input);
      return await this.searchWikipedia(params.query, params.limit);
    } catch (error) {
      throw new Error(
        `Failed to execute Wikipedia tool: ${
          error instanceof Error ? error.message : 'Invalid input'
        }`
      );
    }
  }
}
