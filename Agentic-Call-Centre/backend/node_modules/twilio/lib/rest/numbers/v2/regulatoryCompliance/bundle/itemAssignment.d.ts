/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2 from "../../../V2";
/**
 * Options to pass to create a ItemAssignmentInstance
 */
export interface ItemAssignmentListInstanceCreateOptions {
    /** The SID of an object bag that holds information of the different items. */
    objectSid: string;
}
/**
 * Options to pass to each
 */
export interface ItemAssignmentListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ItemAssignmentInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ItemAssignmentListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ItemAssignmentListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ItemAssignmentContext {
    /**
     * Remove a ItemAssignmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ItemAssignmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ItemAssignmentInstance
     */
    fetch(callback?: (error: Error | null, item?: ItemAssignmentInstance) => any): Promise<ItemAssignmentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ItemAssignmentContextSolution {
    bundleSid: string;
    sid: string;
}
export declare class ItemAssignmentContextImpl implements ItemAssignmentContext {
    protected _version: V2;
    protected _solution: ItemAssignmentContextSolution;
    protected _uri: string;
    constructor(_version: V2, bundleSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ItemAssignmentInstance) => any): Promise<ItemAssignmentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ItemAssignmentContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ItemAssignmentPayload extends TwilioResponsePayload {
    results: ItemAssignmentResource[];
}
interface ItemAssignmentResource {
    sid: string;
    bundle_sid: string;
    account_sid: string;
    object_sid: string;
    date_created: Date;
    url: string;
}
export declare class ItemAssignmentInstance {
    protected _version: V2;
    protected _solution: ItemAssignmentContextSolution;
    protected _context?: ItemAssignmentContext;
    constructor(_version: V2, payload: ItemAssignmentResource, bundleSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Item Assignment resource.
     */
    sid: string;
    /**
     * The unique string that we created to identify the Bundle resource.
     */
    bundleSid: string;
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
     * Remove a ItemAssignmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ItemAssignmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ItemAssignmentInstance
     */
    fetch(callback?: (error: Error | null, item?: ItemAssignmentInstance) => any): Promise<ItemAssignmentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        bundleSid: string;
        accountSid: string;
        objectSid: string;
        dateCreated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ItemAssignmentSolution {
    bundleSid: string;
}
export interface ItemAssignmentListInstance {
    _version: V2;
    _solution: ItemAssignmentSolution;
    _uri: string;
    (sid: string): ItemAssignmentContext;
    get(sid: string): ItemAssignmentContext;
    /**
     * Create a ItemAssignmentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ItemAssignmentInstance
     */
    create(params: ItemAssignmentListInstanceCreateOptions, callback?: (error: Error | null, item?: ItemAssignmentInstance) => any): Promise<ItemAssignmentInstance>;
    /**
     * Streams ItemAssignmentInstance records from the API.
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
     * @param { ItemAssignmentListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ItemAssignmentInstance, done: (err?: Error) => void) => void): void;
    each(params: ItemAssignmentListInstanceEachOptions, callback?: (item: ItemAssignmentInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ItemAssignmentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ItemAssignmentPage) => any): Promise<ItemAssignmentPage>;
    /**
     * Lists ItemAssignmentInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ItemAssignmentListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ItemAssignmentInstance[]) => any): Promise<ItemAssignmentInstance[]>;
    list(params: ItemAssignmentListInstanceOptions, callback?: (error: Error | null, items: ItemAssignmentInstance[]) => any): Promise<ItemAssignmentInstance[]>;
    /**
     * Retrieve a single page of ItemAssignmentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ItemAssignmentListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ItemAssignmentPage) => any): Promise<ItemAssignmentPage>;
    page(params: ItemAssignmentListInstancePageOptions, callback?: (error: Error | null, items: ItemAssignmentPage) => any): Promise<ItemAssignmentPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ItemAssignmentListInstance(version: V2, bundleSid: string): ItemAssignmentListInstance;
export declare class ItemAssignmentPage extends Page<V2, ItemAssignmentPayload, ItemAssignmentResource, ItemAssignmentInstance> {
    /**
     * Initialize the ItemAssignmentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: ItemAssignmentSolution);
    /**
     * Build an instance of ItemAssignmentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ItemAssignmentResource): ItemAssignmentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
