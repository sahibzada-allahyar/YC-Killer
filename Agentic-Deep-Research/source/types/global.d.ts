/// <reference types="node" />

declare module 'zod' {
  export const object: any;
  export const string: any;
  export const array: any;
  export const z: any;
}

declare module 'ai' {
  export interface GenerateObjectOptions<T> {
    model: any;
    system: string;
    prompt: string;
    schema: any;
    abortSignal?: AbortSignal;
  }

  export interface GenerateObjectResult<T> {
    object: T;
  }

  export function generateObject<T>(
    options: GenerateObjectOptions<T>,
  ): Promise<GenerateObjectResult<T>>;
}

declare module '@mendable/firecrawl-js' {
  export interface SearchOptions {
    timeout?: number;
    limit?: number;
    scrapeOptions?: {
      formats?: string[];
    };
  }

  export interface SearchResponseItem {
    url: string;
    markdown: string;
  }

  export interface SearchResponse {
    data: SearchResponseItem[];
  }

  export default class FirecrawlApp {
    constructor(options: { apiKey: string; apiUrl?: string });
    search(query: string, options?: SearchOptions): Promise<SearchResponse>;
  }
}

declare module 'p-limit' {
  type LimitFunction = <T>(fn: () => Promise<T>) => Promise<T>;
  export default function pLimit(concurrency: number): LimitFunction;
}

declare module 'lodash-es' {
  export function compact<T>(array: Array<T | null | undefined>): T[];
}

declare module 'js-tiktoken' {
  export function getEncoding(encoding: string): {
    encode(text: string): number[];
  };
}

declare global {
  namespace NodeJS {
    interface ProcessEnv {
      FIRECRAWL_KEY?: string;
      FIRECRAWL_BASE_URL?: string;
      OPENAI_KEY?: string;
      OPENAI_ENDPOINT?: string;
      OPENAI_MODEL?: string;
      CONTEXT_SIZE?: string;
    }
  }
}
