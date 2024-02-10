/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to update a AccountSecretInstance
 */
export interface AccountSecretContextUpdateOptions {
    /** The secret value; up to 4096 characters. */
    value: string;
}
/**
 * Options to pass to create a AccountSecretInstance
 */
export interface AccountSecretListInstanceCreateOptions {
    /** The secret key; up to 100 characters. */
    key: string;
    /** The secret value; up to 4096 characters. */
    value: string;
}
/**
 * Options to pass to each
 */
export interface AccountSecretListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AccountSecretInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AccountSecretListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AccountSecretListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AccountSecretContext {
    /**
     * Remove a AccountSecretInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AccountSecretInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountSecretInstance
     */
    fetch(callback?: (error: Error | null, item?: AccountSecretInstance) => any): Promise<AccountSecretInstance>;
    /**
     * Update a AccountSecretInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountSecretInstance
     */
    update(params: AccountSecretContextUpdateOptions, callback?: (error: Error | null, item?: AccountSecretInstance) => any): Promise<AccountSecretInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AccountSecretContextSolution {
    key: string;
}
export declare class AccountSecretContextImpl implements AccountSecretContext {
    protected _version: V1;
    protected _solution: AccountSecretContextSolution;
    protected _uri: string;
    constructor(_version: V1, key: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AccountSecretInstance) => any): Promise<AccountSecretInstance>;
    update(params: AccountSecretContextUpdateOptions, callback?: (error: Error | null, item?: AccountSecretInstance) => any): Promise<AccountSecretInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AccountSecretContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AccountSecretPayload extends TwilioResponsePayload {
    secrets: AccountSecretResource[];
}
interface AccountSecretResource {
    key: string;
    date_rotated: Date;
    url: string;
}
export declare class AccountSecretInstance {
    protected _version: V1;
    protected _solution: AccountSecretContextSolution;
    protected _context?: AccountSecretContext;
    constructor(_version: V1, payload: AccountSecretResource, key?: string);
    /**
     * The secret key; up to 100 characters.
     */
    key: string;
    dateRotated: Date;
    /**
     * The absolute URL of the Secret.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a AccountSecretInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AccountSecretInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountSecretInstance
     */
    fetch(callback?: (error: Error | null, item?: AccountSecretInstance) => any): Promise<AccountSecretInstance>;
    /**
     * Update a AccountSecretInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountSecretInstance
     */
    update(params: AccountSecretContextUpdateOptions, callback?: (error: Error | null, item?: AccountSecretInstance) => any): Promise<AccountSecretInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        key: string;
        dateRotated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AccountSecretSolution {
}
export interface AccountSecretListInstance {
    _version: V1;
    _solution: AccountSecretSolution;
    _uri: string;
    (key: string): AccountSecretContext;
    get(key: string): AccountSecretContext;
    /**
     * Create a AccountSecretInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AccountSecretInstance
     */
    create(params: AccountSecretListInstanceCreateOptions, callback?: (error: Error | null, item?: AccountSecretInstance) => any): Promise<AccountSecretInstance>;
    /**
     * Streams AccountSecretInstance records from the API.
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
     * @param { AccountSecretListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AccountSecretInstance, done: (err?: Error) => void) => void): void;
    each(params: AccountSecretListInstanceEachOptions, callback?: (item: AccountSecretInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AccountSecretInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AccountSecretPage) => any): Promise<AccountSecretPage>;
    /**
     * Lists AccountSecretInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AccountSecretListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AccountSecretInstance[]) => any): Promise<AccountSecretInstance[]>;
    list(params: AccountSecretListInstanceOptions, callback?: (error: Error | null, items: AccountSecretInstance[]) => any): Promise<AccountSecretInstance[]>;
    /**
     * Retrieve a single page of AccountSecretInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AccountSecretListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AccountSecretPage) => any): Promise<AccountSecretPage>;
    page(params: AccountSecretListInstancePageOptions, callback?: (error: Error | null, items: AccountSecretPage) => any): Promise<AccountSecretPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AccountSecretListInstance(version: V1): AccountSecretListInstance;
export declare class AccountSecretPage extends Page<V1, AccountSecretPayload, AccountSecretResource, AccountSecretInstance> {
    /**
     * Initialize the AccountSecretPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: AccountSecretSolution);
    /**
     * Build an instance of AccountSecretInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AccountSecretResource): AccountSecretInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
