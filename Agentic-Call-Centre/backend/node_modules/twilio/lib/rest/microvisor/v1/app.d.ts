/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { AppManifestListInstance } from "./app/appManifest";
/**
 * Options to pass to each
 */
export interface AppListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AppInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AppListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AppListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AppContext {
    appManifests: AppManifestListInstance;
    /**
     * Remove a AppInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AppInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AppInstance
     */
    fetch(callback?: (error: Error | null, item?: AppInstance) => any): Promise<AppInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AppContextSolution {
    sid: string;
}
export declare class AppContextImpl implements AppContext {
    protected _version: V1;
    protected _solution: AppContextSolution;
    protected _uri: string;
    protected _appManifests?: AppManifestListInstance;
    constructor(_version: V1, sid: string);
    get appManifests(): AppManifestListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AppInstance) => any): Promise<AppInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AppContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AppPayload extends TwilioResponsePayload {
    apps: AppResource[];
}
interface AppResource {
    sid: string;
    account_sid: string;
    hash: string;
    unique_name: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class AppInstance {
    protected _version: V1;
    protected _solution: AppContextSolution;
    protected _context?: AppContext;
    constructor(_version: V1, payload: AppResource, sid?: string);
    /**
     * A 34-character string that uniquely identifies this App.
     */
    sid: string;
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * App manifest hash represented as `hash_algorithm:hash_value`.
     */
    hash: string;
    /**
     * A developer-defined string that uniquely identifies the App. This value must be unique for all Apps on this Account. The `unique_name` value may be used as an alternative to the `sid` in the URL path to address the resource.
     */
    uniqueName: string;
    /**
     * The date that this App was created, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date that this App was last updated, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The URL of this resource.
     */
    url: string;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a AppInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AppInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AppInstance
     */
    fetch(callback?: (error: Error | null, item?: AppInstance) => any): Promise<AppInstance>;
    /**
     * Access the appManifests.
     */
    appManifests(): AppManifestListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        hash: string;
        uniqueName: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AppSolution {
}
export interface AppListInstance {
    _version: V1;
    _solution: AppSolution;
    _uri: string;
    (sid: string): AppContext;
    get(sid: string): AppContext;
    /**
     * Streams AppInstance records from the API.
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
     * @param { AppListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AppInstance, done: (err?: Error) => void) => void): void;
    each(params: AppListInstanceEachOptions, callback?: (item: AppInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AppInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AppPage) => any): Promise<AppPage>;
    /**
     * Lists AppInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AppListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AppInstance[]) => any): Promise<AppInstance[]>;
    list(params: AppListInstanceOptions, callback?: (error: Error | null, items: AppInstance[]) => any): Promise<AppInstance[]>;
    /**
     * Retrieve a single page of AppInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AppListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AppPage) => any): Promise<AppPage>;
    page(params: AppListInstancePageOptions, callback?: (error: Error | null, items: AppPage) => any): Promise<AppPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AppListInstance(version: V1): AppListInstance;
export declare class AppPage extends Page<V1, AppPayload, AppResource, AppInstance> {
    /**
     * Initialize the AppPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: AppSolution);
    /**
     * Build an instance of AppInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AppResource): AppInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
