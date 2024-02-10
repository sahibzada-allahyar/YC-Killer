import type { LocalPattern } from './image-config';
export declare function matchLocalPattern(pattern: LocalPattern, url: URL): boolean;
export declare function hasLocalMatch(localPatterns: LocalPattern[] | undefined, urlPathAndQuery: string): boolean;
