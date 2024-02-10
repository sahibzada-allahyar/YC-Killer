/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
export type SyncMapItemQueryFromBoundType = "inclusive" | "exclusive";
export type SyncMapItemQueryResultOrder = "asc" | "desc";
/**
 * Options to pass to remove a SyncMapItemInstance
 */
export interface SyncMapItemContextRemoveOptions {
    /** If provided, applies this mutation if (and only if) the “revision” field of this [map item] matches the provided value. This matches the semantics of (and is implemented with) the HTTP [If-Match header](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/If-Match). */
    ifMatch?: string;
}
/**
 * Options to pass to update a SyncMapItemInstance
 */
export interface SyncMapItemContextUpdateOptions {
    /** If provided, applies this mutation if (and only if) the “revision” field of this [map item] matches the provided value. This matches the semantics of (and is implemented with) the HTTP [If-Match header](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/If-Match). */
    ifMatch?: string;
    /** A JSON string that represents an arbitrary, schema-less object that the Map Item stores. Can be up to 16 KiB in length. */
    data?: any;
    /** An alias for `item_ttl`. If both parameters are provided, this value is ignored. */
    ttl?: number;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the Map Item expires (time-to-live) and is deleted. */
    itemTtl?: number;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the Map Item\\\'s parent Sync Map expires (time-to-live) and is deleted. This parameter can only be used when the Map Item\\\'s `data` or `ttl` is updated in the same request. */
    collectionTtl?: number;
}
/**
 * Options to pass to create a SyncMapItemInstance
 */
export interface SyncMapItemListInstanceCreateOptions {
    /** The unique, user-defined key for the Map Item. Can be up to 320 characters long. */
    key: string;
    /** A JSON string that represents an arbitrary, schema-less object that the Map Item stores. Can be up to 16 KiB in length. */
    data: any;
    /** An alias for `item_ttl`. If both parameters are provided, this value is ignored. */
    ttl?: number;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the Map Item expires (time-to-live) and is deleted. */
    itemTtl?: number;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the Map Item\\\'s parent Sync Map expires (time-to-live) and is deleted. */
    collectionTtl?: number;
}
/**
 * Options to pass to each
 */
export interface SyncMapItemListInstanceEachOptions {
    /** How to order the Map Items returned by their `key` value. Can be: `asc` (ascending) or `desc` (descending) and the default is ascending. Map Items are [ordered lexicographically](https://en.wikipedia.org/wiki/Lexicographical_order) by Item key. */
    order?: SyncMapItemQueryResultOrder;
    /** The `key` of the first Sync Map Item resource to read. See also `bounds`. */
    from?: string;
    /** Whether to include the Map Item referenced by the `from` parameter. Can be: `inclusive` to include the Map Item referenced by the `from` parameter or `exclusive` to start with the next Map Item. The default value is `inclusive`. */
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
    /** How to order the Map Items returned by their `key` value. Can be: `asc` (ascending) or `desc` (descending) and the default is ascending. Map Items are [ordered lexicographically](https://en.wikipedia.org/wiki/Lexicographical_order) by Item key. */
    order?: SyncMapItemQueryResultOrder;
    /** The `key` of the first Sync Map Item resource to read. See also `bounds`. */
    from?: string;
    /** Whether to include the Map Item referenced by the `from` parameter. Can be: `inclusive` to include the Map Item referenced by the `from` parameter or `exclusive` to start with the next Map Item. The default value is `inclusive`. */
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
    /** How to order the Map Items returned by their `key` value. Can be: `asc` (ascending) or `desc` (descending) and the default is ascending. Map Items are [ordered lexicographically](https://en.wikipedia.org/wiki/Lexicographical_order) by Item key. */
    order?: SyncMapItemQueryResultOrder;
    /** The `key` of the first Sync Map Item resource to read. See also `bounds`. */
    from?: string;
    /** Whether to include the Map Item referenced by the `from` parameter. Can be: `inclusive` to include the Map Item referenced by the `from` parameter or `exclusive` to start with the next Map Item. The default value is `inclusive`. */
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
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapItemInstance
     */
    update(callback?: (error: Error | null, item?: SyncMapItemInstance) => any): Promise<SyncMapItemInstance>;
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
    protected _version: V1;
    protected _solution: SyncMapItemContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, mapSid: string, key: string);
    remove(params?: SyncMapItemContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SyncMapItemInstance) => any): Promise<SyncMapItemInstance>;
    update(params?: SyncMapItemContextUpdateOptions | ((error: Error | null, item?: SyncMapItemInstance) => any), callback?: (error: Error | null, item?: SyncMapItemInstance) => any): Promise<SyncMapItemInstance>;
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
    date_expires: Date;
    date_created: Date;
    date_updated: Date;
    created_by: string;
}
export declare class SyncMapItemInstance {
    protected _version: V1;
    protected _solution: SyncMapItemContextSolution;
    protected _context?: SyncMapItemContext;
    constructor(_version: V1, payload: SyncMapItemResource, serviceSid: string, mapSid: string, key?: string);
    /**
     * The unique, user-defined key for the Map Item.
     */
    key: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Map Item resource.
     */
    accountSid: string;
    /**
     * The SID of the [Sync Service](https://www.twilio.com/docs/sync/api/service) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the Sync Map that contains the Map Item.
     */
    mapSid: string;
    /**
     * The absolute URL of the Map Item resource.
     */
    url: string;
    /**
     * The current revision of the Map Item, represented as a string.
     */
    revision: string;
    /**
     * An arbitrary, schema-less object that the Map Item stores. Can be up to 16 KiB in length.
     */
    data: any;
    /**
     * The date and time in GMT when the Map Item expires and will be deleted, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. If the Map Item does not expire, this value is `null`.  The Map Item might not be deleted immediately after it expires.
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
     * The identity of the Map Item\'s creator. If the Map Item is created from the client SDK, the value matches the Access Token\'s `identity` field. If the Map Item was created from the REST API, the value is `system`.
     */
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
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapItemInstance
     */
    update(callback?: (error: Error | null, item?: SyncMapItemInstance) => any): Promise<SyncMapItemInstance>;
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
        dateExpires: Date;
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
    _version: V1;
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
export declare function SyncMapItemListInstance(version: V1, serviceSid: string, mapSid: string): SyncMapItemListInstance;
export declare class SyncMapItemPage extends Page<V1, SyncMapItemPayload, SyncMapItemResource, SyncMapItemInstance> {
    /**
     * Initialize the SyncMapItemPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SyncMapItemSolution);
    /**
     * Build an instance of SyncMapItemInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SyncMapItemResource): SyncMapItemInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
