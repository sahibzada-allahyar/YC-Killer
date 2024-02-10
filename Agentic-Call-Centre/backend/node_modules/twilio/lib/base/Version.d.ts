import Domain from "./Domain";
import { RequestOpts } from "./BaseTwilio";
export interface PageLimitOptions {
    /**
     * The maximum number of items to fetch
     */
    limit: number;
    /**
     * The maximum number of items to return with every request
     */
    pageSize: number;
}
export interface PageLimit {
    limit: number;
    pageSize: number;
}
export default class Version {
    _domain: Domain;
    _version: Version | string;
    /**
     *
     * Base version object
     *
     * @param domain - twilio domain
     * @param version - api version
     */
    constructor(domain: Domain, version: string | Version);
    get domain(): Domain;
    /**
     * Generate absolute url from a uri
     *
     * @param uri - uri to transform
     * @returns transformed url
     */
    absoluteUrl(uri: string): string;
    /**
     * Generate relative url from a uri
     *
     * @param uri - uri to transform
     * @returns transformed url
     */
    relativeUrl(uri: string): string;
    /**
     * Make a request against the domain
     *
     * @param opts - request options
     * @returns promise that resolves to request response
     */
    request(opts: RequestOpts): Promise<any>;
    /**
     * Create a new record
     *
     * @param opts - request options
     *
     * @throws Error If response returns non 2xx or 201 status code
     *
     * @returns promise that resolves to created record
     */
    create(opts: RequestOpts): Promise<any>;
    /**
     * Fetch an instance of a record
     *
     * @param opts - request options
     *
     * @throws Error If response returns non 2xx or 3xx status code
     *
     * @returns promise that resolves to fetched result
     */
    fetch(opts: RequestOpts): Promise<any>;
    /**
     * Fetch a page of records
     *
     * @param opts - request options
     * @returns promise that resolves to page of records
     */
    page(opts: RequestOpts): Promise<any>;
    /**
     * Update a record
     *
     * @param opts - request options
     *
     * @throws Error If response returns non 2xx status code
     *
     * @returns promise that resolves to updated result
     */
    update(opts: RequestOpts): Promise<any>;
    /**
     * Delete a record
     *
     * @param opts - request options
     *
     * @throws Error If response returns a 5xx status
     *
     * @returns promise that resolves to true if record was deleted
     */
    remove(opts: RequestOpts): Promise<boolean>;
    /**
     * Process limits for list requests
     *
     * @param opts.limit - The maximum number of items to fetch
     * @param opts.pageSize - The maximum number of items to return with every request
     *
     */
    readLimits(opts: PageLimitOptions): PageLimit;
    setPromiseCallback(operationPromise: any, callback: any): Promise<any>;
    /**
     * For each record instance, executes a provided callback function with that
     * instance
     *
     * @param params - Parameters (Optional)
     * @param params.limit - Optional maximum number of record instances to
     *  fetch
     * @param params.pageSize - Optional maximum number of records to return
     *  with every request
     * @param params.callback - Callback function to call with each
     *  record instance
     * @param params.done - Optional done function to call when all
     *  records are processed, the limit is reached, or an error occurs.
     *  Receives an error argument if an error occurs.
     * @param callback - Callback function to call with each record.
     *  Receives a done function argument that will short-circuit the for-each
     *  loop that may accept an error argument.
     * @returns Returns a promise that resolves when all records
     *  processed or if the limit is reached, and rejects with an error if an
     *  error occurs and is not handled in the user provided done function.
     */
    each<T>(params?: any, callback?: (item: T, done: (err?: Error) => void) => void): Promise<void>;
    list<T>(params?: any, callback?: (error: Error | null, items: T) => any): Promise<any>;
}
