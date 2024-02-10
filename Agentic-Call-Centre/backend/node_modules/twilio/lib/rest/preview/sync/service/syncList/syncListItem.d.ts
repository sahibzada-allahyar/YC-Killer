/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import Sync from "../../../Sync";
export type SyncListItemQueryFromBoundType = "inclusive" | "exclusive";
export type SyncListItemQueryResultOrder = "asc" | "desc";
/**
 * Options to pass to remove a SyncListItemInstance
 */
export interface SyncListItemContextRemoveOptions {
    /** The If-Match HTTP request header */
    ifMatch?: string;
}
/**
 * Options to pass to update a SyncListItemInstance
 */
export interface SyncListItemContextUpdateOptions {
    /**  */
    data: any;
    /** The If-Match HTTP request header */
    ifMatch?: string;
}
/**
 * Options to pass to create a SyncListItemInstance
 */
export interface SyncListItemListInstanceCreateOptions {
    /**  */
    data: any;
}
/**
 * Options to pass to each
 */
export interface SyncListItemListInstanceEachOptions {
    /**  */
    order?: SyncListItemQueryResultOrder;
    /**  */
    from?: string;
    /**  */
    bounds?: SyncListItemQueryFromBoundType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SyncListItemInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SyncListItemListInstanceOptions {
    /**  */
    order?: SyncListItemQueryResultOrder;
    /**  */
    from?: string;
    /**  */
    bounds?: SyncListItemQueryFromBoundType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SyncListItemListInstancePageOptions {
    /**  */
    order?: SyncListItemQueryResultOrder;
    /**  */
    from?: string;
    /**  */
    bounds?: SyncListItemQueryFromBoundType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SyncListItemContext {
    /**
     * Remove a SyncListItemInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a SyncListItemInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListItemInstance
     */
    remove(params: SyncListItemContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SyncListItemInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListItemInstance
     */
    fetch(callback?: (error: Error | null, item?: SyncListItemInstance) => any): Promise<SyncListItemInstance>;
    /**
     * Update a SyncListItemInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListItemInstance
     */
    update(params: SyncListItemContextUpdateOptions, callback?: (error: Error | null, item?: SyncListItemInstance) => any): Promise<SyncListItemInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SyncListItemContextSolution {
    serviceSid: string;
    listSid: string;
    index: number;
}
export declare class SyncListItemContextImpl implements SyncListItemContext {
    protected _version: Sync;
    protected _solution: SyncListItemContextSolution;
    protected _uri: string;
    constructor(_version: Sync, serviceSid: string, listSid: string, index: number);
    remove(params?: SyncListItemContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SyncListItemInstance) => any): Promise<SyncListItemInstance>;
    update(params: SyncListItemContextUpdateOptions, callback?: (error: Error | null, item?: SyncListItemInstance) => any): Promise<SyncListItemInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SyncListItemContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SyncListItemPayload extends TwilioResponsePayload {
    items: SyncListItemResource[];
}
interface SyncListItemResource {
    index: number;
    account_sid: string;
    service_sid: string;
    list_sid: string;
    url: string;
    revision: string;
    data: any;
    date_created: Date;
    date_updated: Date;
    created_by: string;
}
export declare class SyncListItemInstance {
    protected _version: Sync;
    protected _solution: SyncListItemContextSolution;
    protected _context?: SyncListItemContext;
    constructor(_version: Sync, payload: SyncListItemResource, serviceSid: string, listSid: string, index?: number);
    index: number;
    accountSid: string;
    serviceSid: string;
    listSid: string;
    url: string;
    revision: string;
    data: any;
    dateCreated: Date;
    dateUpdated: Date;
    createdBy: string;
    private get _proxy();
    /**
     * Remove a SyncListItemInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a SyncListItemInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListItemInstance
     */
    remove(params: SyncListItemContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SyncListItemInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListItemInstance
     */
    fetch(callback?: (error: Error | null, item?: SyncListItemInstance) => any): Promise<SyncListItemInstance>;
    /**
     * Update a SyncListItemInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListItemInstance
     */
    update(params: SyncListItemContextUpdateOptions, callback?: (error: Error | null, item?: SyncListItemInstance) => any): Promise<SyncListItemInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        index: number;
        accountSid: string;
        serviceSid: string;
        listSid: string;
        url: string;
        revision: string;
        data: any;
        dateCreated: Date;
        dateUpdated: Date;
        createdBy: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SyncListItemSolution {
    serviceSid: string;
    listSid: string;
}
export interface SyncListItemListInstance {
    _version: Sync;
    _solution: SyncListItemSolution;
    _uri: string;
    (index: number): SyncListItemContext;
    get(index: number): SyncListItemContext;
    /**
     * Create a SyncListItemInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListItemInstance
     */
    create(params: SyncListItemListInstanceCreateOptions, callback?: (error: Error | null, item?: SyncListItemInstance) => any): Promise<SyncListItemInstance>;
    /**
     * Streams SyncListItemInstance records from the API.
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
     * @param { SyncListItemListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SyncListItemInstance, done: (err?: Error) => void) => void): void;
    each(params: SyncListItemListInstanceEachOptions, callback?: (item: SyncListItemInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SyncListItemInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SyncListItemPage) => any): Promise<SyncListItemPage>;
    /**
     * Lists SyncListItemInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SyncListItemListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SyncListItemInstance[]) => any): Promise<SyncListItemInstance[]>;
    list(params: SyncListItemListInstanceOptions, callback?: (error: Error | null, items: SyncListItemInstance[]) => any): Promise<SyncListItemInstance[]>;
    /**
     * Retrieve a single page of SyncListItemInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SyncListItemListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SyncListItemPage) => any): Promise<SyncListItemPage>;
    page(params: SyncListItemListInstancePageOptions, callback?: (error: Error | null, items: SyncListItemPage) => any): Promise<SyncListItemPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SyncListItemListInstance(version: Sync, serviceSid: string, listSid: string): SyncListItemListInstance;
export declare class SyncListItemPage extends Page<Sync, SyncListItemPayload, SyncListItemResource, SyncListItemInstance> {
    /**
     * Initialize the SyncListItemPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: Sync, response: Response<string>, solution: SyncListItemSolution);
    /**
     * Build an instance of SyncListItemInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SyncListItemResource): SyncListItemInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
