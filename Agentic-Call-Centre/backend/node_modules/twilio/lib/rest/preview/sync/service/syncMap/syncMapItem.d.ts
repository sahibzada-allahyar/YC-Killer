/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import Sync from "../../../Sync";
export type SyncMapItemQueryFromBoundType = "inclusive" | "exclusive";
export type SyncMapItemQueryResultOrder = "asc" | "desc";
/**
 * Options to pass to remove a SyncMapItemInstance
 */
export interface SyncMapItemContextRemoveOptions {
    /** The If-Match HTTP request header */
    ifMatch?: string;
}
/**
 * Options to pass to update a SyncMapItemInstance
 */
export interface SyncMapItemContextUpdateOptions {
    /**  */
    data: any;
    /** The If-Match HTTP request header */
    ifMatch?: string;
}
/**
 * Options to pass to create a SyncMapItemInstance
 */
export interface SyncMapItemListInstanceCreateOptions {
    /**  */
    key: string;
    /**  */
    data: any;
}
/**
 * Options to pass to each
 */
export interface SyncMapItemListInstanceEachOptions {
    /**  */
    order?: SyncMapItemQueryResultOrder;
    /**  */
    from?: string;
    /**  */
    bounds?: SyncMapItemQueryFromBoundType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SyncMapItemInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SyncMapItemListInstanceOptions {
    /**  */
    order?: SyncMapItemQueryResultOrder;
    /**  */
    from?: string;
    /**  */
    bounds?: SyncMapItemQueryFromBoundType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SyncMapItemListInstancePageOptions {
    /**  */
    order?: SyncMapItemQueryResultOrder;
    /**  */
    from?: string;
    /**  */
    bounds?: SyncMapItemQueryFromBoundType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SyncMapItemContext {
    /**
     * Remove a SyncMapItemInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a SyncMapItemInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapItemInstance
     */
    remove(params: SyncMapItemContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SyncMapItemInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapItemInstance
     */
    fetch(callback?: (error: Error | null, item?: SyncMapItemInstance) => any): Promise<SyncMapItemInstance>;
    /**
     * Update a SyncMapItemInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapItemInstance
     */
    update(params: SyncMapItemContextUpdateOptions, callback?: (error: Error | null, item?: SyncMapItemInstance) => any): Promise<SyncMapItemInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SyncMapItemContextSolution {
    serviceSid: string;
    mapSid: string;
    key: string;
}
export declare class SyncMapItemContextImpl implements SyncMapItemContext {
    protected _version: Sync;
    protected _solution: SyncMapItemContextSolution;
    protected _uri: string;
    constructor(_version: Sync, serviceSid: string, mapSid: string, key: string);
    remove(params?: SyncMapItemContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SyncMapItemInstance) => any): Promise<SyncMapItemInstance>;
    update(params: SyncMapItemContextUpdateOptions, callback?: (error: Error | null, item?: SyncMapItemInstance) => any): Promise<SyncMapItemInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SyncMapItemContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SyncMapItemPayload extends TwilioResponsePayload {
    items: SyncMapItemResource[];
}
interface SyncMapItemResource {
    key: string;
    account_sid: string;
    service_sid: string;
    map_sid: string;
    url: string;
    revision: string;
    data: any;
    date_created: Date;
    date_updated: Date;
    created_by: string;
}
export declare class SyncMapItemInstance {
    protected _version: Sync;
    protected _solution: SyncMapItemContextSolution;
    protected _context?: SyncMapItemContext;
    constructor(_version: Sync, payload: SyncMapItemResource, serviceSid: string, mapSid: string, key?: string);
    key: string;
    accountSid: string;
    serviceSid: string;
    mapSid: string;
    url: string;
    revision: string;
    data: any;
    dateCreated: Date;
    dateUpdated: Date;
    createdBy: string;
    private get _proxy();
    /**
     * Remove a SyncMapItemInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a SyncMapItemInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapItemInstance
     */
    remove(params: SyncMapItemContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SyncMapItemInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapItemInstance
     */
    fetch(callback?: (error: Error | null, item?: SyncMapItemInstance) => any): Promise<SyncMapItemInstance>;
    /**
     * Update a SyncMapItemInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapItemInstance
     */
    update(params: SyncMapItemContextUpdateOptions, callback?: (error: Error | null, item?: SyncMapItemInstance) => any): Promise<SyncMapItemInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        key: string;
        accountSid: string;
        serviceSid: string;
        mapSid: string;
        url: string;
        revision: string;
        data: any;
        dateCreated: Date;
        dateUpdated: Date;
        createdBy: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SyncMapItemSolution {
    serviceSid: string;
    mapSid: string;
}
export interface SyncMapItemListInstance {
    _version: Sync;
    _solution: SyncMapItemSolution;
    _uri: string;
    (key: string): SyncMapItemContext;
    get(key: string): SyncMapItemContext;
    /**
     * Create a SyncMapItemInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapItemInstance
     */
    create(params: SyncMapItemListInstanceCreateOptions, callback?: (error: Error | null, item?: SyncMapItemInstance) => any): Promise<SyncMapItemInstance>;
    /**
     * Streams SyncMapItemInstance records from the API.
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
     * @param { SyncMapItemListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SyncMapItemInstance, done: (err?: Error) => void) => void): void;
    each(params: SyncMapItemListInstanceEachOptions, callback?: (item: SyncMapItemInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SyncMapItemInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SyncMapItemPage) => any): Promise<SyncMapItemPage>;
    /**
     * Lists SyncMapItemInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SyncMapItemListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SyncMapItemInstance[]) => any): Promise<SyncMapItemInstance[]>;
    list(params: SyncMapItemListInstanceOptions, callback?: (error: Error | null, items: SyncMapItemInstance[]) => any): Promise<SyncMapItemInstance[]>;
    /**
     * Retrieve a single page of SyncMapItemInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SyncMapItemListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SyncMapItemPage) => any): Promise<SyncMapItemPage>;
    page(params: SyncMapItemListInstancePageOptions, callback?: (error: Error | null, items: SyncMapItemPage) => any): Promise<SyncMapItemPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SyncMapItemListInstance(version: Sync, serviceSid: string, mapSid: string): SyncMapItemListInstance;
export declare class SyncMapItemPage extends Page<Sync, SyncMapItemPayload, SyncMapItemResource, SyncMapItemInstance> {
    /**
     * Initialize the SyncMapItemPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: Sync, response: Response<string>, solution: SyncMapItemSolution);
    /**
     * Build an instance of SyncMapItemInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SyncMapItemResource): SyncMapItemInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
