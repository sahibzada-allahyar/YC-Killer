/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to update a OriginationUrlInstance
 */
export interface OriginationUrlContextUpdateOptions {
    /** The value that determines the relative share of the load the URI should receive compared to other URIs with the same priority. Can be an integer from 1 to 65535, inclusive, and the default is 10. URLs with higher values receive more load than those with lower ones with the same priority. */
    weight?: number;
    /** The relative importance of the URI. Can be an integer from 0 to 65535, inclusive, and the default is 10. The lowest number represents the most important URI. */
    priority?: number;
    /** Whether the URL is enabled. The default is `true`. */
    enabled?: boolean;
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** The SIP address you want Twilio to route your Origination calls to. This must be a `sip:` schema. `sips` is NOT supported. */
    sipUrl?: string;
}
/**
 * Options to pass to create a OriginationUrlInstance
 */
export interface OriginationUrlListInstanceCreateOptions {
    /** The value that determines the relative share of the load the URI should receive compared to other URIs with the same priority. Can be an integer from 1 to 65535, inclusive, and the default is 10. URLs with higher values receive more load than those with lower ones with the same priority. */
    weight: number;
    /** The relative importance of the URI. Can be an integer from 0 to 65535, inclusive, and the default is 10. The lowest number represents the most important URI. */
    priority: number;
    /** Whether the URL is enabled. The default is `true`. */
    enabled: boolean;
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName: string;
    /** The SIP address you want Twilio to route your Origination calls to. This must be a `sip:` schema. */
    sipUrl: string;
}
/**
 * Options to pass to each
 */
export interface OriginationUrlListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: OriginationUrlInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface OriginationUrlListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface OriginationUrlListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface OriginationUrlContext {
    /**
     * Remove a OriginationUrlInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a OriginationUrlInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OriginationUrlInstance
     */
    fetch(callback?: (error: Error | null, item?: OriginationUrlInstance) => any): Promise<OriginationUrlInstance>;
    /**
     * Update a OriginationUrlInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OriginationUrlInstance
     */
    update(callback?: (error: Error | null, item?: OriginationUrlInstance) => any): Promise<OriginationUrlInstance>;
    /**
     * Update a OriginationUrlInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OriginationUrlInstance
     */
    update(params: OriginationUrlContextUpdateOptions, callback?: (error: Error | null, item?: OriginationUrlInstance) => any): Promise<OriginationUrlInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface OriginationUrlContextSolution {
    trunkSid: string;
    sid: string;
}
export declare class OriginationUrlContextImpl implements OriginationUrlContext {
    protected _version: V1;
    protected _solution: OriginationUrlContextSolution;
    protected _uri: string;
    constructor(_version: V1, trunkSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: OriginationUrlInstance) => any): Promise<OriginationUrlInstance>;
    update(params?: OriginationUrlContextUpdateOptions | ((error: Error | null, item?: OriginationUrlInstance) => any), callback?: (error: Error | null, item?: OriginationUrlInstance) => any): Promise<OriginationUrlInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): OriginationUrlContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface OriginationUrlPayload extends TwilioResponsePayload {
    origination_urls: OriginationUrlResource[];
}
interface OriginationUrlResource {
    account_sid: string;
    sid: string;
    trunk_sid: string;
    weight: number;
    enabled: boolean;
    sip_url: string;
    friendly_name: string;
    priority: number;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class OriginationUrlInstance {
    protected _version: V1;
    protected _solution: OriginationUrlContextSolution;
    protected _context?: OriginationUrlContext;
    constructor(_version: V1, payload: OriginationUrlResource, trunkSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the OriginationUrl resource.
     */
    accountSid: string;
    /**
     * The unique string that we created to identify the OriginationUrl resource.
     */
    sid: string;
    /**
     * The SID of the Trunk that owns the Origination URL.
     */
    trunkSid: string;
    /**
     * The value that determines the relative share of the load the URI should receive compared to other URIs with the same priority. Can be an integer from 1 to 65535, inclusive, and the default is 10. URLs with higher values receive more load than those with lower ones with the same priority.
     */
    weight: number;
    /**
     * Whether the URL is enabled. The default is `true`.
     */
    enabled: boolean;
    /**
     * The SIP address you want Twilio to route your Origination calls to. This must be a `sip:` schema.
     */
    sipUrl: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The relative importance of the URI. Can be an integer from 0 to 65535, inclusive, and the default is 10. The lowest number represents the most important URI.
     */
    priority: number;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a OriginationUrlInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a OriginationUrlInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OriginationUrlInstance
     */
    fetch(callback?: (error: Error | null, item?: OriginationUrlInstance) => any): Promise<OriginationUrlInstance>;
    /**
     * Update a OriginationUrlInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OriginationUrlInstance
     */
    update(callback?: (error: Error | null, item?: OriginationUrlInstance) => any): Promise<OriginationUrlInstance>;
    /**
     * Update a OriginationUrlInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OriginationUrlInstance
     */
    update(params: OriginationUrlContextUpdateOptions, callback?: (error: Error | null, item?: OriginationUrlInstance) => any): Promise<OriginationUrlInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        sid: string;
        trunkSid: string;
        weight: number;
        enabled: boolean;
        sipUrl: string;
        friendlyName: string;
        priority: number;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface OriginationUrlSolution {
    trunkSid: string;
}
export interface OriginationUrlListInstance {
    _version: V1;
    _solution: OriginationUrlSolution;
    _uri: string;
    (sid: string): OriginationUrlContext;
    get(sid: string): OriginationUrlContext;
    /**
     * Create a OriginationUrlInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OriginationUrlInstance
     */
    create(params: OriginationUrlListInstanceCreateOptions, callback?: (error: Error | null, item?: OriginationUrlInstance) => any): Promise<OriginationUrlInstance>;
    /**
     * Streams OriginationUrlInstance records from the API.
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
     * @param { OriginationUrlListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: OriginationUrlInstance, done: (err?: Error) => void) => void): void;
    each(params: OriginationUrlListInstanceEachOptions, callback?: (item: OriginationUrlInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of OriginationUrlInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: OriginationUrlPage) => any): Promise<OriginationUrlPage>;
    /**
     * Lists OriginationUrlInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { OriginationUrlListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: OriginationUrlInstance[]) => any): Promise<OriginationUrlInstance[]>;
    list(params: OriginationUrlListInstanceOptions, callback?: (error: Error | null, items: OriginationUrlInstance[]) => any): Promise<OriginationUrlInstance[]>;
    /**
     * Retrieve a single page of OriginationUrlInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { OriginationUrlListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: OriginationUrlPage) => any): Promise<OriginationUrlPage>;
    page(params: OriginationUrlListInstancePageOptions, callback?: (error: Error | null, items: OriginationUrlPage) => any): Promise<OriginationUrlPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function OriginationUrlListInstance(version: V1, trunkSid: string): OriginationUrlListInstance;
export declare class OriginationUrlPage extends Page<V1, OriginationUrlPayload, OriginationUrlResource, OriginationUrlInstance> {
    /**
     * Initialize the OriginationUrlPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: OriginationUrlSolution);
    /**
     * Build an instance of OriginationUrlInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: OriginationUrlResource): OriginationUrlInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
