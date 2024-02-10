/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { SyncListItemListInstance } from "./syncList/syncListItem";
import { SyncListPermissionListInstance } from "./syncList/syncListPermission";
/**
 * Options to pass to update a SyncListInstance
 */
export interface SyncListContextUpdateOptions {
    /** An alias for `collection_ttl`. If both are provided, this value is ignored. */
    ttl?: number;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the Sync List expires (time-to-live) and is deleted. */
    collectionTtl?: number;
}
/**
 * Options to pass to create a SyncListInstance
 */
export interface SyncListListInstanceCreateOptions {
    /** An application-defined string that uniquely identifies the resource. This value must be unique within its Service and it can be up to 320 characters long. The `unique_name` value can be used as an alternative to the `sid` in the URL path to address the resource. */
    uniqueName?: string;
    /** Alias for collection_ttl. If both are provided, this value is ignored. */
    ttl?: number;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the Sync List expires (time-to-live) and is deleted. */
    collectionTtl?: number;
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
     * Update a SyncListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListInstance
     */
    update(callback?: (error: Error | null, item?: SyncListInstance) => any): Promise<SyncListInstance>;
    /**
     * Update a SyncListInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListInstance
     */
    update(params: SyncListContextUpdateOptions, callback?: (error: Error | null, item?: SyncListInstance) => any): Promise<SyncListInstance>;
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
    protected _version: V1;
    protected _solution: SyncListContextSolution;
    protected _uri: string;
    protected _syncListItems?: SyncListItemListInstance;
    protected _syncListPermissions?: SyncListPermissionListInstance;
    constructor(_version: V1, serviceSid: string, sid: string);
    get syncListItems(): SyncListItemListInstance;
    get syncListPermissions(): SyncListPermissionListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SyncListInstance) => any): Promise<SyncListInstance>;
    update(params?: SyncListContextUpdateOptions | ((error: Error | null, item?: SyncListInstance) => any), callback?: (error: Error | null, item?: SyncListInstance) => any): Promise<SyncListInstance>;
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
    date_expires: Date;
    date_created: Date;
    date_updated: Date;
    created_by: string;
}
export declare class SyncListInstance {
    protected _version: V1;
    protected _solution: SyncListContextSolution;
    protected _context?: SyncListContext;
    constructor(_version: V1, payload: SyncListResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Sync List resource.
     */
    sid: string;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used in place of the resource\'s `sid` in the URL to address the resource.
     */
    uniqueName: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Sync List resource.
     */
    accountSid: string;
    /**
     * The SID of the [Sync Service](https://www.twilio.com/docs/sync/api/service) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The absolute URL of the Sync List resource.
     */
    url: string;
    /**
     * The URLs of the Sync List\'s nested resources.
     */
    links: Record<string, string>;
    /**
     * The current revision of the Sync List, represented as a string.
     */
    revision: string;
    /**
     * The date and time in GMT when the Sync List expires and will be deleted, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. If the Sync List does not expire, this value is `null`. The Sync List might not be deleted immediately after it expires.
     */
    dateExpires: Date;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The identity of the Sync List\'s creator. If the Sync List is created from the client SDK, the value matches the Access Token\'s `identity` field. If the Sync List was created from the REST API, the value is `system`.
     */
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
     * Update a SyncListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListInstance
     */
    update(callback?: (error: Error | null, item?: SyncListInstance) => any): Promise<SyncListInstance>;
    /**
     * Update a SyncListInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListInstance
     */
    update(params: SyncListContextUpdateOptions, callback?: (error: Error | null, item?: SyncListInstance) => any): Promise<SyncListInstance>;
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
        dateExpires: Date;
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
    _version: V1;
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
export declare function SyncListListInstance(version: V1, serviceSid: string): SyncListListInstance;
export declare class SyncListPage extends Page<V1, SyncListPayload, SyncListResource, SyncListInstance> {
    /**
     * Initialize the SyncListPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SyncListSolution);
    /**
     * Build an instance of SyncListInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SyncListResource): SyncListInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
