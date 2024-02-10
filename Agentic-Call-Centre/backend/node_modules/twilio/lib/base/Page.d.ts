import Version from "./Version";
import Response from "../http/response";
export interface TwilioResponsePayload {
    [key: string]: any;
    first_page_uri: string;
    next_page_uri: string;
    page: number;
    page_size: number;
    previous_page_uri: string;
    uri: string;
    meta?: {
        key?: string;
        next_page_url?: string;
        previous_page_url?: string;
    };
}
interface Solution {
    [name: string]: any;
}
export default class Page<TVersion extends Version, TPayload extends TwilioResponsePayload, TResource, TInstance> {
    nextPageUrl?: string;
    previousPageUrl?: string;
    instances: TInstance[];
    _version: TVersion;
    _payload: TPayload;
    _solution: Solution;
    /**
     *
     * Base page object to maintain request state.
     *
     * @param version - A twilio version instance
     * @param response - The http response
     * @param solution - path solution
     */
    constructor(version: TVersion, response: Response<string | TPayload>, solution: Solution);
    /**
     * Meta keys returned in a list request
     *
     * @constant META_KEYS
     */
    static META_KEYS: string[];
    /**
     * Get the url of the previous page of records
     *
     * @returns url of the previous page, or undefined if the
     * previous page URI/URL is not defined.
     */
    getPreviousPageUrl(): string | undefined;
    /**
     * Get the url of the next page of records
     *
     * @returns url of the next page, or undefined if the
     * next page URI/URL is not defined.
     */
    getNextPageUrl(): string | undefined;
    /**
     * Build a new instance given a json payload
     *
     * @param payload - Payload response from the API
     * @returns instance of a resource
     */
    getInstance(payload: any): TInstance;
    /**
     * Load a list of records
     *
     * @param resources - json payload of records
     * @returns list of resources
     */
    loadInstances(resources: TResource[]): TInstance[];
    /**
     * Fetch the next page of records
     *
     * @returns promise that resolves to next page of results,
     * or undefined if there isn't a nextPageUrl undefined.
     */
    nextPage(): Promise<Page<TVersion, TPayload, TResource, TInstance>> | undefined;
    /**
     * Fetch the previous page of records
     *
     * @returns promise that resolves to previous page of
     * results, or undefined if there isn't a previousPageUrl undefined.
     */
    previousPage(): Promise<Page<TVersion, TPayload, TResource, TInstance>> | undefined;
    /**
     * Parse json response from API
     *
     * @param response - API response
     *
     * @throws Error If non 200 status code is returned
     *
     * @returns json parsed response
     */
    processResponse(response: Response<string | TPayload>): TPayload;
    /**
     * Load a page of records
     *
     * @param  {object} payload json payload
     * @return {array} the page of records
     */
    loadPage(payload: TPayload): TResource[];
    forOwn(obj: object, iteratee: (val: any, key: string, object: object) => void): void;
    toJSON(): object;
}
export {};
