/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
/**
 * Options to pass to update a SigningKeyInstance
 */
export interface SigningKeyContextUpdateOptions {
    /**  */
    friendlyName?: string;
}
/**
 * Options to pass to each
 */
export interface SigningKeyListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SigningKeyInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SigningKeyListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SigningKeyListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SigningKeyContext {
    /**
     * Remove a SigningKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SigningKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SigningKeyInstance
     */
    fetch(callback?: (error: Error | null, item?: SigningKeyInstance) => any): Promise<SigningKeyInstance>;
    /**
     * Update a SigningKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SigningKeyInstance
     */
    update(callback?: (error: Error | null, item?: SigningKeyInstance) => any): Promise<SigningKeyInstance>;
    /**
     * Update a SigningKeyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SigningKeyInstance
     */
    update(params: SigningKeyContextUpdateOptions, callback?: (error: Error | null, item?: SigningKeyInstance) => any): Promise<SigningKeyInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SigningKeyContextSolution {
    accountSid: string;
    sid: string;
}
export declare class SigningKeyContextImpl implements SigningKeyContext {
    protected _version: V2010;
    protected _solution: SigningKeyContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SigningKeyInstance) => any): Promise<SigningKeyInstance>;
    update(params?: SigningKeyContextUpdateOptions | ((error: Error | null, item?: SigningKeyInstance) => any), callback?: (error: Error | null, item?: SigningKeyInstance) => any): Promise<SigningKeyInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SigningKeyContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SigningKeyPayload extends TwilioResponsePayload {
    signing_keys: SigningKeyResource[];
}
interface SigningKeyResource {
    sid: string;
    friendly_name: string;
    date_created: Date;
    date_updated: Date;
}
export declare class SigningKeyInstance {
    protected _version: V2010;
    protected _solution: SigningKeyContextSolution;
    protected _context?: SigningKeyContext;
    constructor(_version: V2010, payload: SigningKeyResource, accountSid: string, sid?: string);
    sid: string;
    friendlyName: string;
    dateCreated: Date;
    dateUpdated: Date;
    private get _proxy();
    /**
     * Remove a SigningKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SigningKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SigningKeyInstance
     */
    fetch(callback?: (error: Error | null, item?: SigningKeyInstance) => any): Promise<SigningKeyInstance>;
    /**
     * Update a SigningKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SigningKeyInstance
     */
    update(callback?: (error: Error | null, item?: SigningKeyInstance) => any): Promise<SigningKeyInstance>;
    /**
     * Update a SigningKeyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SigningKeyInstance
     */
    update(params: SigningKeyContextUpdateOptions, callback?: (error: Error | null, item?: SigningKeyInstance) => any): Promise<SigningKeyInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        friendlyName: string;
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SigningKeySolution {
    accountSid: string;
}
export interface SigningKeyListInstance {
    _version: V2010;
    _solution: SigningKeySolution;
    _uri: string;
    (sid: string): SigningKeyContext;
    get(sid: string): SigningKeyContext;
    /**
     * Streams SigningKeyInstance records from the API.
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
     * @param { SigningKeyListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SigningKeyInstance, done: (err?: Error) => void) => void): void;
    each(params: SigningKeyListInstanceEachOptions, callback?: (item: SigningKeyInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SigningKeyInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SigningKeyPage) => any): Promise<SigningKeyPage>;
    /**
     * Lists SigningKeyInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SigningKeyListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SigningKeyInstance[]) => any): Promise<SigningKeyInstance[]>;
    list(params: SigningKeyListInstanceOptions, callback?: (error: Error | null, items: SigningKeyInstance[]) => any): Promise<SigningKeyInstance[]>;
    /**
     * Retrieve a single page of SigningKeyInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SigningKeyListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SigningKeyPage) => any): Promise<SigningKeyPage>;
    page(params: SigningKeyListInstancePageOptions, callback?: (error: Error | null, items: SigningKeyPage) => any): Promise<SigningKeyPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SigningKeyListInstance(version: V2010, accountSid: string): SigningKeyListInstance;
export declare class SigningKeyPage extends Page<V2010, SigningKeyPayload, SigningKeyResource, SigningKeyInstance> {
    /**
     * Initialize the SigningKeyPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: SigningKeySolution);
    /**
     * Build an instance of SigningKeyInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SigningKeyResource): SigningKeyInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
