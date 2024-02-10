/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { SyncMapItemListInstance } from "./syncMap/syncMapItem";
import { SyncMapPermissionListInstance } from "./syncMap/syncMapPermission";
/**
 * Options to pass to update a SyncMapInstance
 */
export interface SyncMapContextUpdateOptions {
    /** An alias for `collection_ttl`. If both parameters are provided, this value is ignored. */
    ttl?: number;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the Sync Map expires (time-to-live) and is deleted. */
    collectionTtl?: number;
}
/**
 * Options to pass to create a SyncMapInstance
 */
export interface SyncMapListInstanceCreateOptions {
    /** An application-defined string that uniquely identifies the resource. It can be used as an alternative to the `sid` in the URL path to address the resource. */
    uniqueName?: string;
    /** An alias for `collection_ttl`. If both parameters are provided, this value is ignored. */
    ttl?: number;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the Sync Map expires (time-to-live) and is deleted. */
    collectionTtl?: number;
}
/**
 * Options to pass to each
 */
export interface SyncMapListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SyncMapInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SyncMapListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SyncMapListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SyncMapContext {
    syncMapItems: SyncMapItemListInstance;
    syncMapPermissions: SyncMapPermissionListInstance;
    /**
     * Remove a SyncMapInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SyncMapInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapInstance
     */
    fetch(callback?: (error: Error | null, item?: SyncMapInstance) => any): Promise<SyncMapInstance>;
    /**
     * Update a SyncMapInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapInstance
     */
    update(callback?: (error: Error | null, item?: SyncMapInstance) => any): Promise<SyncMapInstance>;
    /**
     * Update a SyncMapInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapInstance
     */
    update(params: SyncMapContextUpdateOptions, callback?: (error: Error | null, item?: SyncMapInstance) => any): Promise<SyncMapInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SyncMapContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class SyncMapContextImpl implements SyncMapContext {
    protected _version: V1;
    protected _solution: SyncMapContextSolution;
    protected _uri: string;
    protected _syncMapItems?: SyncMapItemListInstance;
    protected _syncMapPermissions?: SyncMapPermissionListInstance;
    constructor(_version: V1, serviceSid: string, sid: string);
    get syncMapItems(): SyncMapItemListInstance;
    get syncMapPermissions(): SyncMapPermissionListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SyncMapInstance) => any): Promise<SyncMapInstance>;
    update(params?: SyncMapContextUpdateOptions | ((error: Error | null, item?: SyncMapInstance) => any), callback?: (error: Error | null, item?: SyncMapInstance) => any): Promise<SyncMapInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SyncMapContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SyncMapPayload extends TwilioResponsePayload {
    maps: SyncMapResource[];
}
interface SyncMapResource {
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
export declare class SyncMapInstance {
    protected _version: V1;
    protected _solution: SyncMapContextSolution;
    protected _context?: SyncMapContext;
    constructor(_version: V1, payload: SyncMapResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Sync Map resource.
     */
    sid: string;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used in place of the resource\'s `sid` in the URL to address the resource.
     */
    uniqueName: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Sync Map resource.
     */
    accountSid: string;
    /**
     * The SID of the [Sync Service](https://www.twilio.com/docs/sync/api/service) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The absolute URL of the Sync Map resource.
     */
    url: string;
    /**
     * The URLs of the Sync Map\'s nested resources.
     */
    links: Record<string, string>;
    /**
     * The current revision of the Sync Map, represented as a string.
     */
    revision: string;
    /**
     * The date and time in GMT when the Sync Map expires and will be deleted, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. If the Sync Map does not expire, this value is `null`. The Sync Map might not be deleted immediately after it expires.
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
     * The identity of the Sync Map\'s creator. If the Sync Map is created from the client SDK, the value matches the Access Token\'s `identity` field. If the Sync Map was created from the REST API, the value is `system`.
     */
    createdBy: string;
    private get _proxy();
    /**
     * Remove a SyncMapInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SyncMapInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapInstance
     */
    fetch(callback?: (error: Error | null, item?: SyncMapInstance) => any): Promise<SyncMapInstance>;
    /**
     * Update a SyncMapInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapInstance
     */
    update(callback?: (error: Error | null, item?: SyncMapInstance) => any): Promise<SyncMapInstance>;
    /**
     * Update a SyncMapInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapInstance
     */
    update(params: SyncMapContextUpdateOptions, callback?: (error: Error | null, item?: SyncMapInstance) => any): Promise<SyncMapInstance>;
    /**
     * Access the syncMapItems.
     */
    syncMapItems(): SyncMapItemListInstance;
    /**
     * Access the syncMapPermissions.
     */
    syncMapPermissions(): SyncMapPermissionListInstance;
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
export interface SyncMapSolution {
    serviceSid: string;
}
export interface SyncMapListInstance {
    _version: V1;
    _solution: SyncMapSolution;
    _uri: string;
    (sid: string): SyncMapContext;
    get(sid: string): SyncMapContext;
    /**
     * Create a SyncMapInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapInstance
     */
    create(callback?: (error: Error | null, item?: SyncMapInstance) => any): Promise<SyncMapInstance>;
    /**
     * Create a SyncMapInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapInstance
     */
    create(params: SyncMapListInstanceCreateOptions, callback?: (error: Error | null, item?: SyncMapInstance) => any): Promise<SyncMapInstance>;
    /**
     * Streams SyncMapInstance records from the API.
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
     * @param { SyncMapListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SyncMapInstance, done: (err?: Error) => void) => void): void;
    each(params: SyncMapListInstanceEachOptions, callback?: (item: SyncMapInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SyncMapInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SyncMapPage) => any): Promise<SyncMapPage>;
    /**
     * Lists SyncMapInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SyncMapListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SyncMapInstance[]) => any): Promise<SyncMapInstance[]>;
    list(params: SyncMapListInstanceOptions, callback?: (error: Error | null, items: SyncMapInstance[]) => any): Promise<SyncMapInstance[]>;
    /**
     * Retrieve a single page of SyncMapInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SyncMapListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SyncMapPage) => any): Promise<SyncMapPage>;
    page(params: SyncMapListInstancePageOptions, callback?: (error: Error | null, items: SyncMapPage) => any): Promise<SyncMapPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SyncMapListInstance(version: V1, serviceSid: string): SyncMapListInstance;
export declare class SyncMapPage extends Page<V1, SyncMapPayload, SyncMapResource, SyncMapInstance> {
    /**
     * Initialize the SyncMapPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SyncMapSolution);
    /**
     * Build an instance of SyncMapInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SyncMapResource): SyncMapInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
