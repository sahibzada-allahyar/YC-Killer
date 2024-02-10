/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to create a AlphaSenderInstance
 */
export interface AlphaSenderListInstanceCreateOptions {
    /** The Alphanumeric Sender ID string. Can be up to 11 characters long. Valid characters are A-Z, a-z, 0-9, space, hyphen `-`, plus `+`, underscore `_` and ampersand `&`. This value cannot contain only numbers. */
    alphaSender: string;
}
/**
 * Options to pass to each
 */
export interface AlphaSenderListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AlphaSenderInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AlphaSenderListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AlphaSenderListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AlphaSenderContext {
    /**
     * Remove a AlphaSenderInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AlphaSenderInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AlphaSenderInstance
     */
    fetch(callback?: (error: Error | null, item?: AlphaSenderInstance) => any): Promise<AlphaSenderInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AlphaSenderContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class AlphaSenderContextImpl implements AlphaSenderContext {
    protected _version: V1;
    protected _solution: AlphaSenderContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AlphaSenderInstance) => any): Promise<AlphaSenderInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AlphaSenderContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AlphaSenderPayload extends TwilioResponsePayload {
    alpha_senders: AlphaSenderResource[];
}
interface AlphaSenderResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    date_created: Date;
    date_updated: Date;
    alpha_sender: string;
    capabilities: Array<string>;
    url: string;
}
export declare class AlphaSenderInstance {
    protected _version: V1;
    protected _solution: AlphaSenderContextSolution;
    protected _context?: AlphaSenderContext;
    constructor(_version: V1, payload: AlphaSenderResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the AlphaSender resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the AlphaSender resource.
     */
    accountSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/chat/rest/service-resource) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The Alphanumeric Sender ID string.
     */
    alphaSender: string;
    /**
     * An array of values that describe whether the number can receive calls or messages. Can be: `SMS`.
     */
    capabilities: Array<string>;
    /**
     * The absolute URL of the AlphaSender resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a AlphaSenderInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AlphaSenderInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AlphaSenderInstance
     */
    fetch(callback?: (error: Error | null, item?: AlphaSenderInstance) => any): Promise<AlphaSenderInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        alphaSender: string;
        capabilities: string[];
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AlphaSenderSolution {
    serviceSid: string;
}
export interface AlphaSenderListInstance {
    _version: V1;
    _solution: AlphaSenderSolution;
    _uri: string;
    (sid: string): AlphaSenderContext;
    get(sid: string): AlphaSenderContext;
    /**
     * Create a AlphaSenderInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AlphaSenderInstance
     */
    create(params: AlphaSenderListInstanceCreateOptions, callback?: (error: Error | null, item?: AlphaSenderInstance) => any): Promise<AlphaSenderInstance>;
    /**
     * Streams AlphaSenderInstance records from the API.
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
     * @param { AlphaSenderListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AlphaSenderInstance, done: (err?: Error) => void) => void): void;
    each(params: AlphaSenderListInstanceEachOptions, callback?: (item: AlphaSenderInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AlphaSenderInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AlphaSenderPage) => any): Promise<AlphaSenderPage>;
    /**
     * Lists AlphaSenderInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AlphaSenderListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AlphaSenderInstance[]) => any): Promise<AlphaSenderInstance[]>;
    list(params: AlphaSenderListInstanceOptions, callback?: (error: Error | null, items: AlphaSenderInstance[]) => any): Promise<AlphaSenderInstance[]>;
    /**
     * Retrieve a single page of AlphaSenderInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AlphaSenderListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AlphaSenderPage) => any): Promise<AlphaSenderPage>;
    page(params: AlphaSenderListInstancePageOptions, callback?: (error: Error | null, items: AlphaSenderPage) => any): Promise<AlphaSenderPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AlphaSenderListInstance(version: V1, serviceSid: string): AlphaSenderListInstance;
export declare class AlphaSenderPage extends Page<V1, AlphaSenderPayload, AlphaSenderResource, AlphaSenderInstance> {
    /**
     * Initialize the AlphaSenderPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: AlphaSenderSolution);
    /**
     * Build an instance of AlphaSenderInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AlphaSenderResource): AlphaSenderInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
