/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
export type EndUserType = "individual" | "business";
/**
 * Options to pass to update a EndUserInstance
 */
export interface EndUserContextUpdateOptions {
    /** The string that you assigned to describe the resource. */
    friendlyName?: string;
    /** The set of parameters that are the attributes of the End User resource which are derived End User Types. */
    attributes?: any;
}
/**
 * Options to pass to create a EndUserInstance
 */
export interface EndUserListInstanceCreateOptions {
    /** The string that you assigned to describe the resource. */
    friendlyName: string;
    /**  */
    type: EndUserType;
    /** The set of parameters that are the attributes of the End User resource which are derived End User Types. */
    attributes?: any;
}
/**
 * Options to pass to each
 */
export interface EndUserListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: EndUserInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface EndUserListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface EndUserListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface EndUserContext {
    /**
     * Remove a EndUserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a EndUserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EndUserInstance
     */
    fetch(callback?: (error: Error | null, item?: EndUserInstance) => any): Promise<EndUserInstance>;
    /**
     * Update a EndUserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EndUserInstance
     */
    update(callback?: (error: Error | null, item?: EndUserInstance) => any): Promise<EndUserInstance>;
    /**
     * Update a EndUserInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EndUserInstance
     */
    update(params: EndUserContextUpdateOptions, callback?: (error: Error | null, item?: EndUserInstance) => any): Promise<EndUserInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface EndUserContextSolution {
    sid: string;
}
export declare class EndUserContextImpl implements EndUserContext {
    protected _version: V2;
    protected _solution: EndUserContextSolution;
    protected _uri: string;
    constructor(_version: V2, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: EndUserInstance) => any): Promise<EndUserInstance>;
    update(params?: EndUserContextUpdateOptions | ((error: Error | null, item?: EndUserInstance) => any), callback?: (error: Error | null, item?: EndUserInstance) => any): Promise<EndUserInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): EndUserContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface EndUserPayload extends TwilioResponsePayload {
    results: EndUserResource[];
}
interface EndUserResource {
    sid: string;
    account_sid: string;
    friendly_name: string;
    type: EndUserType;
    attributes: any;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class EndUserInstance {
    protected _version: V2;
    protected _solution: EndUserContextSolution;
    protected _context?: EndUserContext;
    constructor(_version: V2, payload: EndUserResource, sid?: string);
    /**
     * The unique string created by Twilio to identify the End User resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the End User resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    type: EndUserType;
    /**
     * The set of parameters that are the attributes of the End Users resource which are listed in the End User Types.
     */
    attributes: any;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the End User resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a EndUserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a EndUserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EndUserInstance
     */
    fetch(callback?: (error: Error | null, item?: EndUserInstance) => any): Promise<EndUserInstance>;
    /**
     * Update a EndUserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EndUserInstance
     */
    update(callback?: (error: Error | null, item?: EndUserInstance) => any): Promise<EndUserInstance>;
    /**
     * Update a EndUserInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EndUserInstance
     */
    update(params: EndUserContextUpdateOptions, callback?: (error: Error | null, item?: EndUserInstance) => any): Promise<EndUserInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        friendlyName: string;
        type: EndUserType;
        attributes: any;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface EndUserSolution {
}
export interface EndUserListInstance {
    _version: V2;
    _solution: EndUserSolution;
    _uri: string;
    (sid: string): EndUserContext;
    get(sid: string): EndUserContext;
    /**
     * Create a EndUserInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EndUserInstance
     */
    create(params: EndUserListInstanceCreateOptions, callback?: (error: Error | null, item?: EndUserInstance) => any): Promise<EndUserInstance>;
    /**
     * Streams EndUserInstance records from the API.
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
     * @param { EndUserListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: EndUserInstance, done: (err?: Error) => void) => void): void;
    each(params: EndUserListInstanceEachOptions, callback?: (item: EndUserInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of EndUserInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: EndUserPage) => any): Promise<EndUserPage>;
    /**
     * Lists EndUserInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EndUserListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: EndUserInstance[]) => any): Promise<EndUserInstance[]>;
    list(params: EndUserListInstanceOptions, callback?: (error: Error | null, items: EndUserInstance[]) => any): Promise<EndUserInstance[]>;
    /**
     * Retrieve a single page of EndUserInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EndUserListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: EndUserPage) => any): Promise<EndUserPage>;
    page(params: EndUserListInstancePageOptions, callback?: (error: Error | null, items: EndUserPage) => any): Promise<EndUserPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function EndUserListInstance(version: V2): EndUserListInstance;
export declare class EndUserPage extends Page<V2, EndUserPayload, EndUserResource, EndUserInstance> {
    /**
     * Initialize the EndUserPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: EndUserSolution);
    /**
     * Build an instance of EndUserInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: EndUserResource): EndUserInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
