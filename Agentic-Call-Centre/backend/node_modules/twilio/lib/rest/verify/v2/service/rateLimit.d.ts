/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
import { BucketListInstance } from "./rateLimit/bucket";
/**
 * Options to pass to update a RateLimitInstance
 */
export interface RateLimitContextUpdateOptions {
    /** Description of this Rate Limit */
    description?: string;
}
/**
 * Options to pass to create a RateLimitInstance
 */
export interface RateLimitListInstanceCreateOptions {
    /** Provides a unique and addressable name to be assigned to this Rate Limit, assigned by the developer, to be optionally used in addition to SID. **This value should not contain PII.** */
    uniqueName: string;
    /** Description of this Rate Limit */
    description?: string;
}
/**
 * Options to pass to each
 */
export interface RateLimitListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: RateLimitInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface RateLimitListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface RateLimitListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface RateLimitContext {
    buckets: BucketListInstance;
    /**
     * Remove a RateLimitInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RateLimitInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RateLimitInstance
     */
    fetch(callback?: (error: Error | null, item?: RateLimitInstance) => any): Promise<RateLimitInstance>;
    /**
     * Update a RateLimitInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RateLimitInstance
     */
    update(callback?: (error: Error | null, item?: RateLimitInstance) => any): Promise<RateLimitInstance>;
    /**
     * Update a RateLimitInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RateLimitInstance
     */
    update(params: RateLimitContextUpdateOptions, callback?: (error: Error | null, item?: RateLimitInstance) => any): Promise<RateLimitInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface RateLimitContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class RateLimitContextImpl implements RateLimitContext {
    protected _version: V2;
    protected _solution: RateLimitContextSolution;
    protected _uri: string;
    protected _buckets?: BucketListInstance;
    constructor(_version: V2, serviceSid: string, sid: string);
    get buckets(): BucketListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: RateLimitInstance) => any): Promise<RateLimitInstance>;
    update(params?: RateLimitContextUpdateOptions | ((error: Error | null, item?: RateLimitInstance) => any), callback?: (error: Error | null, item?: RateLimitInstance) => any): Promise<RateLimitInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): RateLimitContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface RateLimitPayload extends TwilioResponsePayload {
    rate_limits: RateLimitResource[];
}
interface RateLimitResource {
    sid: string;
    service_sid: string;
    account_sid: string;
    unique_name: string;
    description: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class RateLimitInstance {
    protected _version: V2;
    protected _solution: RateLimitContextSolution;
    protected _context?: RateLimitContext;
    constructor(_version: V2, payload: RateLimitResource, serviceSid: string, sid?: string);
    /**
     * A 34 character string that uniquely identifies this Rate Limit.
     */
    sid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/verify/api/service) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Rate Limit resource.
     */
    accountSid: string;
    /**
     * Provides a unique and addressable name to be assigned to this Rate Limit, assigned by the developer, to be optionally used in addition to SID. **This value should not contain PII.**
     */
    uniqueName: string;
    /**
     * Description of this Rate Limit
     */
    description: string;
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
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a RateLimitInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RateLimitInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RateLimitInstance
     */
    fetch(callback?: (error: Error | null, item?: RateLimitInstance) => any): Promise<RateLimitInstance>;
    /**
     * Update a RateLimitInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RateLimitInstance
     */
    update(callback?: (error: Error | null, item?: RateLimitInstance) => any): Promise<RateLimitInstance>;
    /**
     * Update a RateLimitInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RateLimitInstance
     */
    update(params: RateLimitContextUpdateOptions, callback?: (error: Error | null, item?: RateLimitInstance) => any): Promise<RateLimitInstance>;
    /**
     * Access the buckets.
     */
    buckets(): BucketListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        serviceSid: string;
        accountSid: string;
        uniqueName: string;
        description: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface RateLimitSolution {
    serviceSid: string;
}
export interface RateLimitListInstance {
    _version: V2;
    _solution: RateLimitSolution;
    _uri: string;
    (sid: string): RateLimitContext;
    get(sid: string): RateLimitContext;
    /**
     * Create a RateLimitInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RateLimitInstance
     */
    create(params: RateLimitListInstanceCreateOptions, callback?: (error: Error | null, item?: RateLimitInstance) => any): Promise<RateLimitInstance>;
    /**
     * Streams RateLimitInstance records from the API.
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
     * @param { RateLimitListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: RateLimitInstance, done: (err?: Error) => void) => void): void;
    each(params: RateLimitListInstanceEachOptions, callback?: (item: RateLimitInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of RateLimitInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: RateLimitPage) => any): Promise<RateLimitPage>;
    /**
     * Lists RateLimitInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RateLimitListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: RateLimitInstance[]) => any): Promise<RateLimitInstance[]>;
    list(params: RateLimitListInstanceOptions, callback?: (error: Error | null, items: RateLimitInstance[]) => any): Promise<RateLimitInstance[]>;
    /**
     * Retrieve a single page of RateLimitInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RateLimitListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: RateLimitPage) => any): Promise<RateLimitPage>;
    page(params: RateLimitListInstancePageOptions, callback?: (error: Error | null, items: RateLimitPage) => any): Promise<RateLimitPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RateLimitListInstance(version: V2, serviceSid: string): RateLimitListInstance;
export declare class RateLimitPage extends Page<V2, RateLimitPayload, RateLimitResource, RateLimitInstance> {
    /**
     * Initialize the RateLimitPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: RateLimitSolution);
    /**
     * Build an instance of RateLimitInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: RateLimitResource): RateLimitInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
