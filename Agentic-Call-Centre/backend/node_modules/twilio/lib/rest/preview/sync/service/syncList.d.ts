/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import Sync from "../../Sync";
import { SyncListItemListInstance } from "./syncList/syncListItem";
import { SyncListPermissionListInstance } from "./syncList/syncListPermission";
/**
 * Options to pass to create a SyncListInstance
 */
export interface SyncListListInstanceCreateOptions {
    /**  */
    uniqueName?: string;
}
/**
 * Options to pass to each
 */
export interface SyncListListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SyncListInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SyncListListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SyncListListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SyncListContext {
    syncListItems: SyncListItemListInstance;
    syncListPermissions: SyncListPermissionListInstance;
    /**
     * Remove a SyncListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SyncListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListInstance
     */
    fetch(callback?: (error: Error | null, item?: SyncListInstance) => any): Promise<SyncListInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SyncListContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class SyncListContextImpl implements SyncListContext {
    protected _version: Sync;
    protected _solution: SyncListContextSolution;
    protected _uri: string;
    protected _syncListItems?: SyncListItemListInstance;
    protected _syncListPermissions?: SyncListPermissionListInstance;
    constructor(_version: Sync, serviceSid: string, sid: string);
    get syncListItems(): SyncListItemListInstance;
    get syncListPermissions(): SyncListPermissionListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SyncListInstance) => any): Promise<SyncListInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SyncListContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SyncListPayload extends TwilioResponsePayload {
    lists: SyncListResource[];
}
interface SyncListResource {
    sid: string;
    unique_name: string;
    account_sid: string;
    service_sid: string;
    url: string;
    links: Record<string, string>;
    revision: string;
    date_created: Date;
    date_updated: Date;
    created_by: string;
}
export declare class SyncListInstance {
    protected _version: Sync;
    protected _solution: SyncListContextSolution;
    protected _context?: SyncListContext;
    constructor(_version: Sync, payload: SyncListResource, serviceSid: string, sid?: string);
    sid: string;
    uniqueName: string;
    accountSid: string;
    serviceSid: string;
    url: string;
    links: Record<string, string>;
    revision: string;
    dateCreated: Date;
    dateUpdated: Date;
    createdBy: string;
    private get _proxy();
    /**
     * Remove a SyncListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SyncListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListInstance
     */
    fetch(callback?: (error: Error | null, item?: SyncListInstance) => any): Promise<SyncListInstance>;
    /**
     * Access the syncListItems.
     */
    syncListItems(): SyncListItemListInstance;
    /**
     * Access the syncListPermissions.
     */
    syncListPermissions(): SyncListPermissionListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        uniqueName: string;
        accountSid: string;
        serviceSid: string;
        url: string;
        links: Record<string, string>;
        revision: string;
        dateCreated: Date;
        dateUpdated: Date;
        createdBy: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SyncListSolution {
    serviceSid: string;
}
export interface SyncListListInstance {
    _version: Sync;
    _solution: SyncListSolution;
    _uri: string;
    (sid: string): SyncListContext;
    get(sid: string): SyncListContext;
    /**
     * Create a SyncListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListInstance
     */
    create(callback?: (error: Error | null, item?: SyncListInstance) => any): Promise<SyncListInstance>;
    /**
     * Create a SyncListInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListInstance
     */
    create(params: SyncListListInstanceCreateOptions, callback?: (error: Error | null, item?: SyncListInstance) => any): Promise<SyncListInstance>;
    /**
     * Streams SyncListInstance records from the API.
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
     * @param { SyncListListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SyncListInstance, done: (err?: Error) => void) => void): void;
    each(params: SyncListListInstanceEachOptions, callback?: (item: SyncListInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SyncListInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SyncListPage) => any): Promise<SyncListPage>;
    /**
     * Lists SyncListInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SyncListListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SyncListInstance[]) => any): Promise<SyncListInstance[]>;
    list(params: SyncListListInstanceOptions, callback?: (error: Error | null, items: SyncListInstance[]) => any): Promise<SyncListInstance[]>;
    /**
     * Retrieve a single page of SyncListInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SyncListListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SyncListPage) => any): Promise<SyncListPage>;
    page(params: SyncListListInstancePageOptions, callback?: (error: Error | null, items: SyncListPage) => any): Promise<SyncListPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SyncListListInstance(version: Sync, serviceSid: string): SyncListListInstance;
export declare class SyncListPage extends Page<Sync, SyncListPayload, SyncListResource, SyncListInstance> {
    /**
     * Initialize the SyncListPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: Sync, response: Response<string>, solution: SyncListSolution);
    /**
     * Build an instance of SyncListInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SyncListResource): SyncListInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
