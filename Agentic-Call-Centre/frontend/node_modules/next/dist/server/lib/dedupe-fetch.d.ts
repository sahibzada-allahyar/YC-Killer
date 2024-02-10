export declare function createDedupeFetch(originalFetch: typeof fetch): (resource: URL | RequestInfo, options?: RequestInit) => Promise<Response>;
