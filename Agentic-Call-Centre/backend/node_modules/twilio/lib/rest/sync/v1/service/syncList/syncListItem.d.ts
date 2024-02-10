/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
export type SyncListItemQueryFromBoundType = "inclusive" | "exclusive";
export type SyncListItemQueryResultOrder = "asc" | "desc";
/**
 * Options to pass to remove a SyncListItemInstance
 */
export interface SyncListItemContextRemoveOptions {
    /** If provided, applies this mutation if (and only if) the “revision” field of this [map item] matches the provided value. This matches the semantics of (and is implemented with) the HTTP [If-Match header](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/If-Match). */
    ifMatch?: string;
}
/**
 * Options to pass to update a SyncListItemInstance
 */
export interface SyncListItemContextUpdateOptions {
    /** If provided, applies this mutation if (and only if) the “revision” field of this [map item] matches the provided value. This matches the semantics of (and is implemented with) the HTTP [If-Match header](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/If-Match). */
    ifMatch?: string;
    /** A JSON string that represents an arbitrary, schema-less object that the List Item stores. Can be up to 16 KiB in length. */
    data?: any;
    /** An alias for `item_ttl`. If both parameters are provided, this value is ignored. */
    ttl?: number;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the List Item expires (time-to-live) and is deleted. */
    itemTtl?: number;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the List Item\\\'s parent Sync List expires (time-to-live) and is deleted. This parameter can only be used when the List Item\\\'s `data` or `ttl` is updated in the same request. */
    collectionTtl?: number;
}
/**
 * Options to pass to create a SyncListItemInstance
 */
export interface SyncListItemListInstanceCreateOptions {
    /** A JSON string that represents an arbitrary, schema-less object that the List Item stores. Can be up to 16 KiB in length. */
    data: any;
    /** An alias for `item_ttl`. If both parameters are provided, this value is ignored. */
    ttl?: number;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the List Item expires (time-to-live) and is deleted. */
    itemTtl?: number;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the List Item\\\'s parent Sync List expires (time-to-live) and is deleted. */
    collectionTtl?: number;
}
/**
 * Options to pass to each
 */
export interface SyncListItemListInstanceEachOptions {
    /** How to order the List Items returned by their `index` value. Can be: `asc` (ascending) or `desc` (descending) and the default is ascending. */
    order?: SyncListItemQueryResultOrder;
    /** The `index` of the first Sync List Item resource to read. See also `bounds`. */
    from?: string;
    /** Whether to include the List Item referenced by the `from` parameter. Can be: `inclusive` to include the List Item referenced by the `from` parameter or `exclusive` to start with the next List Item. The default value is `inclusive`. */
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
    /** How to order the List Items returned by their `index` value. Can be: `asc` (ascending) or `desc` (descending) and the default is ascending. */
    order?: SyncListItemQueryResultOrder;
    /** The `index` of the first Sync List Item resource to read. See also `bounds`. */
    from?: string;
    /** Whether to include the List Item referenced by the `from` parameter. Can be: `inclusive` to include the List Item referenced by the `from` parameter or `exclusive` to start with the next List Item. The default value is `inclusive`. */
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
    /** How to order the List Items returned by their `index` value. Can be: `asc` (ascending) or `desc` (descending) and the default is ascending. */
    order?: SyncListItemQueryResultOrder;
    /** The `index` of the first Sync List Item resource to read. See also `bounds`. */
    from?: string;
    /** Whether to include the List Item referenced by the `from` parameter. Can be: `inclusive` to include the List Item referenced by the `from` parameter or `exclusive` to start with the next List Item. The default value is `inclusive`. */
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
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListItemInstance
     */
    update(callback?: (error: Error | null, item?: SyncListItemInstance) => any): Promise<SyncListItemInstance>;
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
    protected _version: V1;
    protected _solution: SyncListItemContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, listSid: string, index: number);
    remove(params?: SyncListItemContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SyncListItemInstance) => any): Promise<SyncListItemInstance>;
    update(params?: SyncListItemContextUpdateOptions | ((error: Error | null, item?: SyncListItemInstance) => any), callback?: (error: Error | null, item?: SyncListItemInstance) => any): Promise<SyncListItemInstance>;
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
    date_expires: Date;
    date_created: Date;
    date_updated: Date;
    created_by: string;
}
export declare class SyncListItemInstance {
    protected _version: V1;
    protected _solution: SyncListItemContextSolution;
    protected _context?: SyncListItemContext;
    constructor(_version: V1, payload: SyncListItemResource, serviceSid: string, listSid: string, index?: number);
    /**
     * The automatically generated index of the List Item. The `index` values of the List Items in a Sync List can have gaps in their sequence.
     */
    index: number;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the List Item resource.
     */
    accountSid: string;
    /**
     * The SID of the [Sync Service](https://www.twilio.com/docs/sync/api/service) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the Sync List that contains the List Item.
     */
    listSid: string;
    /**
     * The absolute URL of the List Item resource.
     */
    url: string;
    /**
     * The current revision of the item, represented as a string.
     */
    revision: string;
    /**
     * An arbitrary, schema-less object that the List Item stores. Can be up to 16 KiB in length.
     */
    data: any;
    /**
     * The date and time in GMT when the List Item expires and will be deleted, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. If the List Item does not expire, this value is `null`. The List Item resource might not be deleted immediately after it expires.
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
     * The identity of the List Item\'s creator. If the item is created from the client SDK, the value matches the Access Token\'s `identity` field. If the item was created from the REST API, the value is `system`.
     */
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
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncListItemInstance
     */
    update(callback?: (error: Error | null, item?: SyncListItemInstance) => any): Promise<SyncListItemInstance>;
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
        dateExpires: Date;
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
    _version: V1;
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
export declare function SyncListItemListInstance(version: V1, serviceSid: string, listSid: string): SyncListItemListInstance;
export declare class SyncListItemPage extends Page<V1, SyncListItemPayload, SyncListItemResource, SyncListItemInstance> {
    /**
     * Initialize the SyncListItemPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SyncListItemSolution);
    /**
     * Build an instance of SyncListItemInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SyncListItemResource): SyncListItemInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
