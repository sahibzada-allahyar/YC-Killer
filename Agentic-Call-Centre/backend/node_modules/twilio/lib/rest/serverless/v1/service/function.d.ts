/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { FunctionVersionListInstance } from "./function/functionVersion";
/**
 * Options to pass to update a FunctionInstance
 */
export interface FunctionContextUpdateOptions {
    /** A descriptive string that you create to describe the Function resource. It can be a maximum of 255 characters. */
    friendlyName: string;
}
/**
 * Options to pass to create a FunctionInstance
 */
export interface FunctionListInstanceCreateOptions {
    /** A descriptive string that you create to describe the Function resource. It can be a maximum of 255 characters. */
    friendlyName: string;
}
/**
 * Options to pass to each
 */
export interface FunctionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: FunctionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface FunctionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface FunctionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface FunctionContext {
    functionVersions: FunctionVersionListInstance;
    /**
     * Remove a FunctionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a FunctionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FunctionInstance
     */
    fetch(callback?: (error: Error | null, item?: FunctionInstance) => any): Promise<FunctionInstance>;
    /**
     * Update a FunctionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FunctionInstance
     */
    update(params: FunctionContextUpdateOptions, callback?: (error: Error | null, item?: FunctionInstance) => any): Promise<FunctionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface FunctionContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class FunctionContextImpl implements FunctionContext {
    protected _version: V1;
    protected _solution: FunctionContextSolution;
    protected _uri: string;
    protected _functionVersions?: FunctionVersionListInstance;
    constructor(_version: V1, serviceSid: string, sid: string);
    get functionVersions(): FunctionVersionListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: FunctionInstance) => any): Promise<FunctionInstance>;
    update(params: FunctionContextUpdateOptions, callback?: (error: Error | null, item?: FunctionInstance) => any): Promise<FunctionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): FunctionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface FunctionPayload extends TwilioResponsePayload {
    functions: FunctionResource[];
}
interface FunctionResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    friendly_name: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class FunctionInstance {
    protected _version: V1;
    protected _solution: FunctionContextSolution;
    protected _context?: FunctionContext;
    constructor(_version: V1, payload: FunctionResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Function resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Function resource.
     */
    accountSid: string;
    /**
     * The SID of the Service that the Function resource is associated with.
     */
    serviceSid: string;
    /**
     * The string that you assigned to describe the Function resource. It can be a maximum of 255 characters.
     */
    friendlyName: string;
    /**
     * The date and time in GMT when the Function resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the Function resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Function resource.
     */
    url: string;
    /**
     * The URLs of nested resources of the Function resource.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a FunctionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a FunctionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FunctionInstance
     */
    fetch(callback?: (error: Error | null, item?: FunctionInstance) => any): Promise<FunctionInstance>;
    /**
     * Update a FunctionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FunctionInstance
     */
    update(params: FunctionContextUpdateOptions, callback?: (error: Error | null, item?: FunctionInstance) => any): Promise<FunctionInstance>;
    /**
     * Access the functionVersions.
     */
    functionVersions(): FunctionVersionListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        friendlyName: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface FunctionSolution {
    serviceSid: string;
}
export interface FunctionListInstance {
    _version: V1;
    _solution: FunctionSolution;
    _uri: string;
    (sid: string): FunctionContext;
    get(sid: string): FunctionContext;
    /**
     * Create a FunctionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FunctionInstance
     */
    create(params: FunctionListInstanceCreateOptions, callback?: (error: Error | null, item?: FunctionInstance) => any): Promise<FunctionInstance>;
    /**
     * Streams FunctionInstance records from the API.
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
     * @param { FunctionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: FunctionInstance, done: (err?: Error) => void) => void): void;
    each(params: FunctionListInstanceEachOptions, callback?: (item: FunctionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of FunctionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: FunctionPage) => any): Promise<FunctionPage>;
    /**
     * Lists FunctionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FunctionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: FunctionInstance[]) => any): Promise<FunctionInstance[]>;
    list(params: FunctionListInstanceOptions, callback?: (error: Error | null, items: FunctionInstance[]) => any): Promise<FunctionInstance[]>;
    /**
     * Retrieve a single page of FunctionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FunctionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: FunctionPage) => any): Promise<FunctionPage>;
    page(params: FunctionListInstancePageOptions, callback?: (error: Error | null, items: FunctionPage) => any): Promise<FunctionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function FunctionListInstance(version: V1, serviceSid: string): FunctionListInstance;
export declare class FunctionPage extends Page<V1, FunctionPayload, FunctionResource, FunctionInstance> {
    /**
     * Initialize the FunctionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: FunctionSolution);
    /**
     * Build an instance of FunctionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: FunctionResource): FunctionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
