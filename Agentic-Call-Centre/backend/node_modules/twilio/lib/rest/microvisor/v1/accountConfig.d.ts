/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to update a AccountConfigInstance
 */
export interface AccountConfigContextUpdateOptions {
    /** The config value; up to 4096 characters. */
    value: string;
}
/**
 * Options to pass to create a AccountConfigInstance
 */
export interface AccountConfigListInstanceCreateOptions {
    /** The config key; up to 100 characters. */
    key: string;
    /** The config value; up to 4096 characters. */
    value: string;
}
/**
 * Options to pass to each
 */
export interface AccountConfigListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AccountConfigInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AccountConfigListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AccountConfigListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AccountConfigContext {
    /**
     * Remove a AccountConfigInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AccountConfigInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountConfigInstance
     */
    fetch(callback?: (error: Error | null, item?: AccountConfigInstance) => any): Promise<AccountConfigInstance>;
    /**
     * Update a AccountConfigInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountConfigInstance
     */
    update(params: AccountConfigContextUpdateOptions, callback?: (error: Error | null, item?: AccountConfigInstance) => any): Promise<AccountConfigInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AccountConfigContextSolution {
    key: string;
}
export declare class AccountConfigContextImpl implements AccountConfigContext {
    protected _version: V1;
    protected _solution: AccountConfigContextSolution;
    protected _uri: string;
    constructor(_version: V1, key: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AccountConfigInstance) => any): Promise<AccountConfigInstance>;
    update(params: AccountConfigContextUpdateOptions, callback?: (error: Error | null, item?: AccountConfigInstance) => any): Promise<AccountConfigInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AccountConfigContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AccountConfigPayload extends TwilioResponsePayload {
    configs: AccountConfigResource[];
}
interface AccountConfigResource {
    key: string;
    date_updated: Date;
    value: string;
    url: string;
}
export declare class AccountConfigInstance {
    protected _version: V1;
    protected _solution: AccountConfigContextSolution;
    protected _context?: AccountConfigContext;
    constructor(_version: V1, payload: AccountConfigResource, key?: string);
    /**
     * The config key; up to 100 characters.
     */
    key: string;
    dateUpdated: Date;
    /**
     * The config value; up to 4096 characters.
     */
    value: string;
    /**
     * The absolute URL of the Config.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a AccountConfigInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AccountConfigInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountConfigInstance
     */
    fetch(callback?: (error: Error | null, item?: AccountConfigInstance) => any): Promise<AccountConfigInstance>;
    /**
     * Update a AccountConfigInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountConfigInstance
     */
    update(params: AccountConfigContextUpdateOptions, callback?: (error: Error | null, item?: AccountConfigInstance) => any): Promise<AccountConfigInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        key: string;
        dateUpdated: Date;
        value: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AccountConfigSolution {
}
export interface AccountConfigListInstance {
    _version: V1;
    _solution: AccountConfigSolution;
    _uri: string;
    (key: string): AccountConfigContext;
    get(key: string): AccountConfigContext;
    /**
     * Create a AccountConfigInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountConfigInstance
     */
    create(params: AccountConfigListInstanceCreateOptions, callback?: (error: Error | null, item?: AccountConfigInstance) => any): Promise<AccountConfigInstance>;
    /**
     * Streams AccountConfigInstance records from the API.
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
     * @param { AccountConfigListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AccountConfigInstance, done: (err?: Error) => void) => void): void;
    each(params: AccountConfigListInstanceEachOptions, callback?: (item: AccountConfigInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AccountConfigInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AccountConfigPage) => any): Promise<AccountConfigPage>;
    /**
     * Lists AccountConfigInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AccountConfigListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AccountConfigInstance[]) => any): Promise<AccountConfigInstance[]>;
    list(params: AccountConfigListInstanceOptions, callback?: (error: Error | null, items: AccountConfigInstance[]) => any): Promise<AccountConfigInstance[]>;
    /**
     * Retrieve a single page of AccountConfigInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AccountConfigListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AccountConfigPage) => any): Promise<AccountConfigPage>;
    page(params: AccountConfigListInstancePageOptions, callback?: (error: Error | null, items: AccountConfigPage) => any): Promise<AccountConfigPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AccountConfigListInstance(version: V1): AccountConfigListInstance;
export declare class AccountConfigPage extends Page<V1, AccountConfigPayload, AccountConfigResource, AccountConfigInstance> {
    /**
     * Initialize the AccountConfigPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: AccountConfigSolution);
    /**
     * Build an instance of AccountConfigInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AccountConfigResource): AccountConfigInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
