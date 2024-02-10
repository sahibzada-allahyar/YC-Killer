/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
export type AssetVersionVisibility = "public" | "private" | "protected";
/**
 * Options to pass to each
 */
export interface AssetVersionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AssetVersionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AssetVersionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AssetVersionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AssetVersionContext {
    /**
     * Fetch a AssetVersionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssetVersionInstance
     */
    fetch(callback?: (error: Error | null, item?: AssetVersionInstance) => any): Promise<AssetVersionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AssetVersionContextSolution {
    serviceSid: string;
    assetSid: string;
    sid: string;
}
export declare class AssetVersionContextImpl implements AssetVersionContext {
    protected _version: V1;
    protected _solution: AssetVersionContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, assetSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: AssetVersionInstance) => any): Promise<AssetVersionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AssetVersionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AssetVersionPayload extends TwilioResponsePayload {
    asset_versions: AssetVersionResource[];
}
interface AssetVersionResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    asset_sid: string;
    path: string;
    visibility: AssetVersionVisibility;
    date_created: Date;
    url: string;
}
export declare class AssetVersionInstance {
    protected _version: V1;
    protected _solution: AssetVersionContextSolution;
    protected _context?: AssetVersionContext;
    constructor(_version: V1, payload: AssetVersionResource, serviceSid: string, assetSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Asset Version resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Asset Version resource.
     */
    accountSid: string;
    /**
     * The SID of the Service that the Asset Version resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the Asset resource that is the parent of the Asset Version.
     */
    assetSid: string;
    /**
     * The URL-friendly string by which the Asset Version can be referenced. It can be a maximum of 255 characters. All paths begin with a forward slash (\'/\'). If an Asset Version creation request is submitted with a path not containing a leading slash, the path will automatically be prepended with one.
     */
    path: string;
    visibility: AssetVersionVisibility;
    /**
     * The date and time in GMT when the Asset Version resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The absolute URL of the Asset Version resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a AssetVersionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssetVersionInstance
     */
    fetch(callback?: (error: Error | null, item?: AssetVersionInstance) => any): Promise<AssetVersionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        assetSid: string;
        path: string;
        visibility: AssetVersionVisibility;
        dateCreated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AssetVersionSolution {
    serviceSid: string;
    assetSid: string;
}
export interface AssetVersionListInstance {
    _version: V1;
    _solution: AssetVersionSolution;
    _uri: string;
    (sid: string): AssetVersionContext;
    get(sid: string): AssetVersionContext;
    /**
     * Streams AssetVersionInstance records from the API.
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
     * @param { AssetVersionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AssetVersionInstance, done: (err?: Error) => void) => void): void;
    each(params: AssetVersionListInstanceEachOptions, callback?: (item: AssetVersionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AssetVersionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AssetVersionPage) => any): Promise<AssetVersionPage>;
    /**
     * Lists AssetVersionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AssetVersionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AssetVersionInstance[]) => any): Promise<AssetVersionInstance[]>;
    list(params: AssetVersionListInstanceOptions, callback?: (error: Error | null, items: AssetVersionInstance[]) => any): Promise<AssetVersionInstance[]>;
    /**
     * Retrieve a single page of AssetVersionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AssetVersionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AssetVersionPage) => any): Promise<AssetVersionPage>;
    page(params: AssetVersionListInstancePageOptions, callback?: (error: Error | null, items: AssetVersionPage) => any): Promise<AssetVersionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AssetVersionListInstance(version: V1, serviceSid: string, assetSid: string): AssetVersionListInstance;
export declare class AssetVersionPage extends Page<V1, AssetVersionPayload, AssetVersionResource, AssetVersionInstance> {
    /**
     * Initialize the AssetVersionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: AssetVersionSolution);
    /**
     * Build an instance of AssetVersionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AssetVersionResource): AssetVersionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
