/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
export type AuthorizedConnectAppPermission = "get-all" | "post-all";
/**
 * Options to pass to each
 */
export interface AuthorizedConnectAppListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AuthorizedConnectAppInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AuthorizedConnectAppListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AuthorizedConnectAppListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AuthorizedConnectAppContext {
    /**
     * Fetch a AuthorizedConnectAppInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthorizedConnectAppInstance
     */
    fetch(callback?: (error: Error | null, item?: AuthorizedConnectAppInstance) => any): Promise<AuthorizedConnectAppInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AuthorizedConnectAppContextSolution {
    accountSid: string;
    connectAppSid: string;
}
export declare class AuthorizedConnectAppContextImpl implements AuthorizedConnectAppContext {
    protected _version: V2010;
    protected _solution: AuthorizedConnectAppContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, connectAppSid: string);
    fetch(callback?: (error: Error | null, item?: AuthorizedConnectAppInstance) => any): Promise<AuthorizedConnectAppInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AuthorizedConnectAppContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AuthorizedConnectAppPayload extends TwilioResponsePayload {
    authorized_connect_apps: AuthorizedConnectAppResource[];
}
interface AuthorizedConnectAppResource {
    account_sid: string;
    connect_app_company_name: string;
    connect_app_description: string;
    connect_app_friendly_name: string;
    connect_app_homepage_url: string;
    connect_app_sid: string;
    permissions: Array<AuthorizedConnectAppPermission>;
    uri: string;
}
export declare class AuthorizedConnectAppInstance {
    protected _version: V2010;
    protected _solution: AuthorizedConnectAppContextSolution;
    protected _context?: AuthorizedConnectAppContext;
    constructor(_version: V2010, payload: AuthorizedConnectAppResource, accountSid: string, connectAppSid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the AuthorizedConnectApp resource.
     */
    accountSid: string;
    /**
     * The company name set for the Connect App.
     */
    connectAppCompanyName: string;
    /**
     * A detailed description of the Connect App.
     */
    connectAppDescription: string;
    /**
     * The name of the Connect App.
     */
    connectAppFriendlyName: string;
    /**
     * The public URL for the Connect App.
     */
    connectAppHomepageUrl: string;
    /**
     * The SID that we assigned to the Connect App.
     */
    connectAppSid: string;
    /**
     * The set of permissions that you authorized for the Connect App.  Can be: `get-all` or `post-all`.
     */
    permissions: Array<AuthorizedConnectAppPermission>;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Fetch a AuthorizedConnectAppInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthorizedConnectAppInstance
     */
    fetch(callback?: (error: Error | null, item?: AuthorizedConnectAppInstance) => any): Promise<AuthorizedConnectAppInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        connectAppCompanyName: string;
        connectAppDescription: string;
        connectAppFriendlyName: string;
        connectAppHomepageUrl: string;
        connectAppSid: string;
        permissions: AuthorizedConnectAppPermission[];
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AuthorizedConnectAppSolution {
    accountSid: string;
}
export interface AuthorizedConnectAppListInstance {
    _version: V2010;
    _solution: AuthorizedConnectAppSolution;
    _uri: string;
    (connectAppSid: string): AuthorizedConnectAppContext;
    get(connectAppSid: string): AuthorizedConnectAppContext;
    /**
     * Streams AuthorizedConnectAppInstance records from the API.
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
     * @param { AuthorizedConnectAppListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AuthorizedConnectAppInstance, done: (err?: Error) => void) => void): void;
    each(params: AuthorizedConnectAppListInstanceEachOptions, callback?: (item: AuthorizedConnectAppInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AuthorizedConnectAppInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AuthorizedConnectAppPage) => any): Promise<AuthorizedConnectAppPage>;
    /**
     * Lists AuthorizedConnectAppInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AuthorizedConnectAppListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AuthorizedConnectAppInstance[]) => any): Promise<AuthorizedConnectAppInstance[]>;
    list(params: AuthorizedConnectAppListInstanceOptions, callback?: (error: Error | null, items: AuthorizedConnectAppInstance[]) => any): Promise<AuthorizedConnectAppInstance[]>;
    /**
     * Retrieve a single page of AuthorizedConnectAppInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AuthorizedConnectAppListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AuthorizedConnectAppPage) => any): Promise<AuthorizedConnectAppPage>;
    page(params: AuthorizedConnectAppListInstancePageOptions, callback?: (error: Error | null, items: AuthorizedConnectAppPage) => any): Promise<AuthorizedConnectAppPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AuthorizedConnectAppListInstance(version: V2010, accountSid: string): AuthorizedConnectAppListInstance;
export declare class AuthorizedConnectAppPage extends Page<V2010, AuthorizedConnectAppPayload, AuthorizedConnectAppResource, AuthorizedConnectAppInstance> {
    /**
     * Initialize the AuthorizedConnectAppPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: AuthorizedConnectAppSolution);
    /**
     * Build an instance of AuthorizedConnectAppInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AuthorizedConnectAppResource): AuthorizedConnectAppInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
