/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to create a TrustProductsEntityAssignmentsInstance
 */
export interface TrustProductsEntityAssignmentsListInstanceCreateOptions {
    /** The SID of an object bag that holds information of the different items. */
    objectSid: string;
}
/**
 * Options to pass to each
 */
export interface TrustProductsEntityAssignmentsListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TrustProductsEntityAssignmentsInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TrustProductsEntityAssignmentsListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TrustProductsEntityAssignmentsListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TrustProductsEntityAssignmentsContext {
    /**
     * Remove a TrustProductsEntityAssignmentsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TrustProductsEntityAssignmentsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrustProductsEntityAssignmentsInstance
     */
    fetch(callback?: (error: Error | null, item?: TrustProductsEntityAssignmentsInstance) => any): Promise<TrustProductsEntityAssignmentsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TrustProductsEntityAssignmentsContextSolution {
    trustProductSid: string;
    sid: string;
}
export declare class TrustProductsEntityAssignmentsContextImpl implements TrustProductsEntityAssignmentsContext {
    protected _version: V1;
    protected _solution: TrustProductsEntityAssignmentsContextSolution;
    protected _uri: string;
    constructor(_version: V1, trustProductSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: TrustProductsEntityAssignmentsInstance) => any): Promise<TrustProductsEntityAssignmentsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TrustProductsEntityAssignmentsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TrustProductsEntityAssignmentsPayload extends TwilioResponsePayload {
    results: TrustProductsEntityAssignmentsResource[];
}
interface TrustProductsEntityAssignmentsResource {
    sid: string;
    trust_product_sid: string;
    account_sid: string;
    object_sid: string;
    date_created: Date;
    url: string;
}
export declare class TrustProductsEntityAssignmentsInstance {
    protected _version: V1;
    protected _solution: TrustProductsEntityAssignmentsContextSolution;
    protected _context?: TrustProductsEntityAssignmentsContext;
    constructor(_version: V1, payload: TrustProductsEntityAssignmentsResource, trustProductSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Item Assignment resource.
     */
    sid: string;
    /**
     * The unique string that we created to identify the TrustProduct resource.
     */
    trustProductSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Item Assignment resource.
     */
    accountSid: string;
    /**
     * The SID of an object bag that holds information of the different items.
     */
    objectSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The absolute URL of the Identity resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a TrustProductsEntityAssignmentsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TrustProductsEntityAssignmentsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrustProductsEntityAssignmentsInstance
     */
    fetch(callback?: (error: Error | null, item?: TrustProductsEntityAssignmentsInstance) => any): Promise<TrustProductsEntityAssignmentsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        trustProductSid: string;
        accountSid: string;
        objectSid: string;
        dateCreated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TrustProductsEntityAssignmentsSolution {
    trustProductSid: string;
}
export interface TrustProductsEntityAssignmentsListInstance {
    _version: V1;
    _solution: TrustProductsEntityAssignmentsSolution;
    _uri: string;
    (sid: string): TrustProductsEntityAssignmentsContext;
    get(sid: string): TrustProductsEntityAssignmentsContext;
    /**
     * Create a TrustProductsEntityAssignmentsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrustProductsEntityAssignmentsInstance
     */
    create(params: TrustProductsEntityAssignmentsListInstanceCreateOptions, callback?: (error: Error | null, item?: TrustProductsEntityAssignmentsInstance) => any): Promise<TrustProductsEntityAssignmentsInstance>;
    /**
     * Streams TrustProductsEntityAssignmentsInstance records from the API.
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
     * @param { TrustProductsEntityAssignmentsListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TrustProductsEntityAssignmentsInstance, done: (err?: Error) => void) => void): void;
    each(params: TrustProductsEntityAssignmentsListInstanceEachOptions, callback?: (item: TrustProductsEntityAssignmentsInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TrustProductsEntityAssignmentsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TrustProductsEntityAssignmentsPage) => any): Promise<TrustProductsEntityAssignmentsPage>;
    /**
     * Lists TrustProductsEntityAssignmentsInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TrustProductsEntityAssignmentsListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TrustProductsEntityAssignmentsInstance[]) => any): Promise<TrustProductsEntityAssignmentsInstance[]>;
    list(params: TrustProductsEntityAssignmentsListInstanceOptions, callback?: (error: Error | null, items: TrustProductsEntityAssignmentsInstance[]) => any): Promise<TrustProductsEntityAssignmentsInstance[]>;
    /**
     * Retrieve a single page of TrustProductsEntityAssignmentsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TrustProductsEntityAssignmentsListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TrustProductsEntityAssignmentsPage) => any): Promise<TrustProductsEntityAssignmentsPage>;
    page(params: TrustProductsEntityAssignmentsListInstancePageOptions, callback?: (error: Error | null, items: TrustProductsEntityAssignmentsPage) => any): Promise<TrustProductsEntityAssignmentsPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TrustProductsEntityAssignmentsListInstance(version: V1, trustProductSid: string): TrustProductsEntityAssignmentsListInstance;
export declare class TrustProductsEntityAssignmentsPage extends Page<V1, TrustProductsEntityAssignmentsPayload, TrustProductsEntityAssignmentsResource, TrustProductsEntityAssignmentsInstance> {
    /**
     * Initialize the TrustProductsEntityAssignmentsPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: TrustProductsEntityAssignmentsSolution);
    /**
     * Build an instance of TrustProductsEntityAssignmentsInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TrustProductsEntityAssignmentsResource): TrustProductsEntityAssignmentsInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
