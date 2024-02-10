/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2 from "../../../V2";
/**
 * Options to pass to update a BucketInstance
 */
export interface BucketContextUpdateOptions {
    /** Maximum number of requests permitted in during the interval. */
    max?: number;
    /** Number of seconds that the rate limit will be enforced over. */
    interval?: number;
}
/**
 * Options to pass to create a BucketInstance
 */
export interface BucketListInstanceCreateOptions {
    /** Maximum number of requests permitted in during the interval. */
    max: number;
    /** Number of seconds that the rate limit will be enforced over. */
    interval: number;
}
/**
 * Options to pass to each
 */
export interface BucketListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: BucketInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface BucketListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface BucketListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface BucketContext {
    /**
     * Remove a BucketInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a BucketInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BucketInstance
     */
    fetch(callback?: (error: Error | null, item?: BucketInstance) => any): Promise<BucketInstance>;
    /**
     * Update a BucketInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BucketInstance
     */
    update(callback?: (error: Error | null, item?: BucketInstance) => any): Promise<BucketInstance>;
    /**
     * Update a BucketInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BucketInstance
     */
    update(params: BucketContextUpdateOptions, callback?: (error: Error | null, item?: BucketInstance) => any): Promise<BucketInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface BucketContextSolution {
    serviceSid: string;
    rateLimitSid: string;
    sid: string;
}
export declare class BucketContextImpl implements BucketContext {
    protected _version: V2;
    protected _solution: BucketContextSolution;
    protected _uri: string;
    constructor(_version: V2, serviceSid: string, rateLimitSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: BucketInstance) => any): Promise<BucketInstance>;
    update(params?: BucketContextUpdateOptions | ((error: Error | null, item?: BucketInstance) => any), callback?: (error: Error | null, item?: BucketInstance) => any): Promise<BucketInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): BucketContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface BucketPayload extends TwilioResponsePayload {
    buckets: BucketResource[];
}
interface BucketResource {
    sid: string;
    rate_limit_sid: string;
    service_sid: string;
    account_sid: string;
    max: number;
    interval: number;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class BucketInstance {
    protected _version: V2;
    protected _solution: BucketContextSolution;
    protected _context?: BucketContext;
    constructor(_version: V2, payload: BucketResource, serviceSid: string, rateLimitSid: string, sid?: string);
    /**
     * A 34 character string that uniquely identifies this Bucket.
     */
    sid: string;
    /**
     * The Twilio-provided string that uniquely identifies the Rate Limit resource.
     */
    rateLimitSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/verify/api/service) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Rate Limit resource.
     */
    accountSid: string;
    /**
     * Maximum number of requests permitted in during the interval.
     */
    max: number;
    /**
     * Number of seconds that the rate limit will be enforced over.
     */
    interval: number;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The URL of this resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a BucketInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a BucketInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BucketInstance
     */
    fetch(callback?: (error: Error | null, item?: BucketInstance) => any): Promise<BucketInstance>;
    /**
     * Update a BucketInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BucketInstance
     */
    update(callback?: (error: Error | null, item?: BucketInstance) => any): Promise<BucketInstance>;
    /**
     * Update a BucketInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BucketInstance
     */
    update(params: BucketContextUpdateOptions, callback?: (error: Error | null, item?: BucketInstance) => any): Promise<BucketInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        rateLimitSid: string;
        serviceSid: string;
        accountSid: string;
        max: number;
        interval: number;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface BucketSolution {
    serviceSid: string;
    rateLimitSid: string;
}
export interface BucketListInstance {
    _version: V2;
    _solution: BucketSolution;
    _uri: string;
    (sid: string): BucketContext;
    get(sid: string): BucketContext;
    /**
     * Create a BucketInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BucketInstance
     */
    create(params: BucketListInstanceCreateOptions, callback?: (error: Error | null, item?: BucketInstance) => any): Promise<BucketInstance>;
    /**
     * Streams BucketInstance records from the API.
     *
     * This operation lazily loads records as efficiently as possible until the limit
     * is reached.
     *
     * The results are passed into the callback function, so this operation is memory
     * efficient.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BucketListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: BucketInstance, done: (err?: Error) => void) => void): void;
    each(params: BucketListInstanceEachOptions, callback?: (item: BucketInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of BucketInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: BucketPage) => any): Promise<BucketPage>;
    /**
     * Lists BucketInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BucketListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: BucketInstance[]) => any): Promise<BucketInstance[]>;
    list(params: BucketListInstanceOptions, callback?: (error: Error | null, items: BucketInstance[]) => any): Promise<BucketInstance[]>;
    /**
     * Retrieve a single page of BucketInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BucketListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: BucketPage) => any): Promise<BucketPage>;
    page(params: BucketListInstancePageOptions, callback?: (error: Error | null, items: BucketPage) => any): Promise<BucketPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BucketListInstance(version: V2, serviceSid: string, rateLimitSid: string): BucketListInstance;
export declare class BucketPage extends Page<V2, BucketPayload, BucketResource, BucketInstance> {
    /**
     * Initialize the BucketPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: BucketSolution);
    /**
     * Build an instance of BucketInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: BucketResource): BucketInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
