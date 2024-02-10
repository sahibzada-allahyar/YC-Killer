/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { AssetVersionListInstance } from "./asset/assetVersion";
/**
 * Options to pass to update a AssetInstance
 */
export interface AssetContextUpdateOptions {
    /** A descriptive string that you create to describe the Asset resource. It can be a maximum of 255 characters. */
    friendlyName: string;
}
/**
 * Options to pass to create a AssetInstance
 */
export interface AssetListInstanceCreateOptions {
    /** A descriptive string that you create to describe the Asset resource. It can be a maximum of 255 characters. */
    friendlyName: string;
}
/**
 * Options to pass to each
 */
export interface AssetListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AssetInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AssetListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AssetListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AssetContext {
    assetVersions: AssetVersionListInstance;
    /**
     * Remove a AssetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AssetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssetInstance
     */
    fetch(callback?: (error: Error | null, item?: AssetInstance) => any): Promise<AssetInstance>;
    /**
     * Update a AssetInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssetInstance
     */
    update(params: AssetContextUpdateOptions, callback?: (error: Error | null, item?: AssetInstance) => any): Promise<AssetInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AssetContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class AssetContextImpl implements AssetContext {
    protected _version: V1;
    protected _solution: AssetContextSolution;
    protected _uri: string;
    protected _assetVersions?: AssetVersionListInstance;
    constructor(_version: V1, serviceSid: string, sid: string);
    get assetVersions(): AssetVersionListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AssetInstance) => any): Promise<AssetInstance>;
    update(params: AssetContextUpdateOptions, callback?: (error: Error | null, item?: AssetInstance) => any): Promise<AssetInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AssetContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AssetPayload extends TwilioResponsePayload {
    assets: AssetResource[];
}
interface AssetResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    friendly_name: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class AssetInstance {
    protected _version: V1;
    protected _solution: AssetContextSolution;
    protected _context?: AssetContext;
    constructor(_version: V1, payload: AssetResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Asset resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Asset resource.
     */
    accountSid: string;
    /**
     * The SID of the Service that the Asset resource is associated with.
     */
    serviceSid: string;
    /**
     * The string that you assigned to describe the Asset resource. It can be a maximum of 255 characters.
     */
    friendlyName: string;
    /**
     * The date and time in GMT when the Asset resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the Asset resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Asset resource.
     */
    url: string;
    /**
     * The URLs of the Asset resource\'s nested resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a AssetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AssetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssetInstance
     */
    fetch(callback?: (error: Error | null, item?: AssetInstance) => any): Promise<AssetInstance>;
    /**
     * Update a AssetInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssetInstance
     */
    update(params: AssetContextUpdateOptions, callback?: (error: Error | null, item?: AssetInstance) => any): Promise<AssetInstance>;
    /**
     * Access the assetVersions.
     */
    assetVersions(): AssetVersionListInstance;
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
export interface AssetSolution {
    serviceSid: string;
}
export interface AssetListInstance {
    _version: V1;
    _solution: AssetSolution;
    _uri: string;
    (sid: string): AssetContext;
    get(sid: string): AssetContext;
    /**
     * Create a AssetInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssetInstance
     */
    create(params: AssetListInstanceCreateOptions, callback?: (error: Error | null, item?: AssetInstance) => any): Promise<AssetInstance>;
    /**
     * Streams AssetInstance records from the API.
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
     * @param { AssetListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AssetInstance, done: (err?: Error) => void) => void): void;
    each(params: AssetListInstanceEachOptions, callback?: (item: AssetInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AssetInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AssetPage) => any): Promise<AssetPage>;
    /**
     * Lists AssetInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AssetListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AssetInstance[]) => any): Promise<AssetInstance[]>;
    list(params: AssetListInstanceOptions, callback?: (error: Error | null, items: AssetInstance[]) => any): Promise<AssetInstance[]>;
    /**
     * Retrieve a single page of AssetInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AssetListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AssetPage) => any): Promise<AssetPage>;
    page(params: AssetListInstancePageOptions, callback?: (error: Error | null, items: AssetPage) => any): Promise<AssetPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AssetListInstance(version: V1, serviceSid: string): AssetListInstance;
export declare class AssetPage extends Page<V1, AssetPayload, AssetResource, AssetInstance> {
    /**
     * Initialize the AssetPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: AssetSolution);
    /**
     * Build an instance of AssetInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AssetResource): AssetInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
