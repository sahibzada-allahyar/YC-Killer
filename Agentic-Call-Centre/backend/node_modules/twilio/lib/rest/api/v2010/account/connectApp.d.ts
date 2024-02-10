/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
export type ConnectAppPermission = "get-all" | "post-all";
/**
 * Options to pass to update a ConnectAppInstance
 */
export interface ConnectAppContextUpdateOptions {
    /** The URL to redirect the user to after we authenticate the user and obtain authorization to access the Connect App. */
    authorizeRedirectUrl?: string;
    /** The company name to set for the Connect App. */
    companyName?: string;
    /** The HTTP method to use when calling `deauthorize_callback_url`. */
    deauthorizeCallbackMethod?: string;
    /** The URL to call using the `deauthorize_callback_method` to de-authorize the Connect App. */
    deauthorizeCallbackUrl?: string;
    /** A description of the Connect App. */
    description?: string;
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** A public URL where users can obtain more information about this Connect App. */
    homepageUrl?: string;
    /** A comma-separated list of the permissions you will request from the users of this ConnectApp.  Can include: `get-all` and `post-all`. */
    permissions?: Array<ConnectAppPermission>;
}
/**
 * Options to pass to each
 */
export interface ConnectAppListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ConnectAppInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ConnectAppListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ConnectAppListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ConnectAppContext {
    /**
     * Remove a ConnectAppInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ConnectAppInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectAppInstance
     */
    fetch(callback?: (error: Error | null, item?: ConnectAppInstance) => any): Promise<ConnectAppInstance>;
    /**
     * Update a ConnectAppInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectAppInstance
     */
    update(callback?: (error: Error | null, item?: ConnectAppInstance) => any): Promise<ConnectAppInstance>;
    /**
     * Update a ConnectAppInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectAppInstance
     */
    update(params: ConnectAppContextUpdateOptions, callback?: (error: Error | null, item?: ConnectAppInstance) => any): Promise<ConnectAppInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ConnectAppContextSolution {
    accountSid: string;
    sid: string;
}
export declare class ConnectAppContextImpl implements ConnectAppContext {
    protected _version: V2010;
    protected _solution: ConnectAppContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ConnectAppInstance) => any): Promise<ConnectAppInstance>;
    update(params?: ConnectAppContextUpdateOptions | ((error: Error | null, item?: ConnectAppInstance) => any), callback?: (error: Error | null, item?: ConnectAppInstance) => any): Promise<ConnectAppInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ConnectAppContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ConnectAppPayload extends TwilioResponsePayload {
    connect_apps: ConnectAppResource[];
}
interface ConnectAppResource {
    account_sid: string;
    authorize_redirect_url: string;
    company_name: string;
    deauthorize_callback_method: string;
    deauthorize_callback_url: string;
    description: string;
    friendly_name: string;
    homepage_url: string;
    permissions: Array<ConnectAppPermission>;
    sid: string;
    uri: string;
}
export declare class ConnectAppInstance {
    protected _version: V2010;
    protected _solution: ConnectAppContextSolution;
    protected _context?: ConnectAppContext;
    constructor(_version: V2010, payload: ConnectAppResource, accountSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the ConnectApp resource.
     */
    accountSid: string;
    /**
     * The URL we redirect the user to after we authenticate the user and obtain authorization to access the Connect App.
     */
    authorizeRedirectUrl: string;
    /**
     * The company name set for the Connect App.
     */
    companyName: string;
    /**
     * The HTTP method we use to call `deauthorize_callback_url`.
     */
    deauthorizeCallbackMethod: string;
    /**
     * The URL we call using the `deauthorize_callback_method` to de-authorize the Connect App.
     */
    deauthorizeCallbackUrl: string;
    /**
     * The description of the Connect App.
     */
    description: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The public URL where users can obtain more information about this Connect App.
     */
    homepageUrl: string;
    /**
     * The set of permissions that your ConnectApp requests.
     */
    permissions: Array<ConnectAppPermission>;
    /**
     * The unique string that that we created to identify the ConnectApp resource.
     */
    sid: string;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Remove a ConnectAppInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ConnectAppInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectAppInstance
     */
    fetch(callback?: (error: Error | null, item?: ConnectAppInstance) => any): Promise<ConnectAppInstance>;
    /**
     * Update a ConnectAppInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectAppInstance
     */
    update(callback?: (error: Error | null, item?: ConnectAppInstance) => any): Promise<ConnectAppInstance>;
    /**
     * Update a ConnectAppInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectAppInstance
     */
    update(params: ConnectAppContextUpdateOptions, callback?: (error: Error | null, item?: ConnectAppInstance) => any): Promise<ConnectAppInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        authorizeRedirectUrl: string;
        companyName: string;
        deauthorizeCallbackMethod: string;
        deauthorizeCallbackUrl: string;
        description: string;
        friendlyName: string;
        homepageUrl: string;
        permissions: ConnectAppPermission[];
        sid: string;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ConnectAppSolution {
    accountSid: string;
}
export interface ConnectAppListInstance {
    _version: V2010;
    _solution: ConnectAppSolution;
    _uri: string;
    (sid: string): ConnectAppContext;
    get(sid: string): ConnectAppContext;
    /**
     * Streams ConnectAppInstance records from the API.
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
     * @param { ConnectAppListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ConnectAppInstance, done: (err?: Error) => void) => void): void;
    each(params: ConnectAppListInstanceEachOptions, callback?: (item: ConnectAppInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ConnectAppInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ConnectAppPage) => any): Promise<ConnectAppPage>;
    /**
     * Lists ConnectAppInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ConnectAppListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ConnectAppInstance[]) => any): Promise<ConnectAppInstance[]>;
    list(params: ConnectAppListInstanceOptions, callback?: (error: Error | null, items: ConnectAppInstance[]) => any): Promise<ConnectAppInstance[]>;
    /**
     * Retrieve a single page of ConnectAppInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ConnectAppListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ConnectAppPage) => any): Promise<ConnectAppPage>;
    page(params: ConnectAppListInstancePageOptions, callback?: (error: Error | null, items: ConnectAppPage) => any): Promise<ConnectAppPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ConnectAppListInstance(version: V2010, accountSid: string): ConnectAppListInstance;
export declare class ConnectAppPage extends Page<V2010, ConnectAppPayload, ConnectAppResource, ConnectAppInstance> {
    /**
     * Initialize the ConnectAppPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: ConnectAppSolution);
    /**
     * Build an instance of ConnectAppInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ConnectAppResource): ConnectAppInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
