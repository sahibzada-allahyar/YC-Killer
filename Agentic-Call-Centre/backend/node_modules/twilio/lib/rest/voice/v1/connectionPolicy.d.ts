/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { ConnectionPolicyTargetListInstance } from "./connectionPolicy/connectionPolicyTarget";
/**
 * Options to pass to update a ConnectionPolicyInstance
 */
export interface ConnectionPolicyContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It is not unique and can be up to 255 characters long. */
    friendlyName?: string;
}
/**
 * Options to pass to create a ConnectionPolicyInstance
 */
export interface ConnectionPolicyListInstanceCreateOptions {
    /** A descriptive string that you create to describe the resource. It is not unique and can be up to 255 characters long. */
    friendlyName?: string;
}
/**
 * Options to pass to each
 */
export interface ConnectionPolicyListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ConnectionPolicyInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ConnectionPolicyListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ConnectionPolicyListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ConnectionPolicyContext {
    targets: ConnectionPolicyTargetListInstance;
    /**
     * Remove a ConnectionPolicyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ConnectionPolicyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyInstance
     */
    fetch(callback?: (error: Error | null, item?: ConnectionPolicyInstance) => any): Promise<ConnectionPolicyInstance>;
    /**
     * Update a ConnectionPolicyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyInstance
     */
    update(callback?: (error: Error | null, item?: ConnectionPolicyInstance) => any): Promise<ConnectionPolicyInstance>;
    /**
     * Update a ConnectionPolicyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyInstance
     */
    update(params: ConnectionPolicyContextUpdateOptions, callback?: (error: Error | null, item?: ConnectionPolicyInstance) => any): Promise<ConnectionPolicyInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ConnectionPolicyContextSolution {
    sid: string;
}
export declare class ConnectionPolicyContextImpl implements ConnectionPolicyContext {
    protected _version: V1;
    protected _solution: ConnectionPolicyContextSolution;
    protected _uri: string;
    protected _targets?: ConnectionPolicyTargetListInstance;
    constructor(_version: V1, sid: string);
    get targets(): ConnectionPolicyTargetListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ConnectionPolicyInstance) => any): Promise<ConnectionPolicyInstance>;
    update(params?: ConnectionPolicyContextUpdateOptions | ((error: Error | null, item?: ConnectionPolicyInstance) => any), callback?: (error: Error | null, item?: ConnectionPolicyInstance) => any): Promise<ConnectionPolicyInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ConnectionPolicyContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ConnectionPolicyPayload extends TwilioResponsePayload {
    connection_policies: ConnectionPolicyResource[];
}
interface ConnectionPolicyResource {
    account_sid: string;
    sid: string;
    friendly_name: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class ConnectionPolicyInstance {
    protected _version: V1;
    protected _solution: ConnectionPolicyContextSolution;
    protected _context?: ConnectionPolicyContext;
    constructor(_version: V1, payload: ConnectionPolicyResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Connection Policy resource.
     */
    accountSid: string;
    /**
     * The unique string that we created to identify the Connection Policy resource.
     */
    sid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a ConnectionPolicyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ConnectionPolicyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyInstance
     */
    fetch(callback?: (error: Error | null, item?: ConnectionPolicyInstance) => any): Promise<ConnectionPolicyInstance>;
    /**
     * Update a ConnectionPolicyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyInstance
     */
    update(callback?: (error: Error | null, item?: ConnectionPolicyInstance) => any): Promise<ConnectionPolicyInstance>;
    /**
     * Update a ConnectionPolicyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyInstance
     */
    update(params: ConnectionPolicyContextUpdateOptions, callback?: (error: Error | null, item?: ConnectionPolicyInstance) => any): Promise<ConnectionPolicyInstance>;
    /**
     * Access the targets.
     */
    targets(): ConnectionPolicyTargetListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        sid: string;
        friendlyName: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ConnectionPolicySolution {
}
export interface ConnectionPolicyListInstance {
    _version: V1;
    _solution: ConnectionPolicySolution;
    _uri: string;
    (sid: string): ConnectionPolicyContext;
    get(sid: string): ConnectionPolicyContext;
    /**
     * Create a ConnectionPolicyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyInstance
     */
    create(callback?: (error: Error | null, item?: ConnectionPolicyInstance) => any): Promise<ConnectionPolicyInstance>;
    /**
     * Create a ConnectionPolicyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyInstance
     */
    create(params: ConnectionPolicyListInstanceCreateOptions, callback?: (error: Error | null, item?: ConnectionPolicyInstance) => any): Promise<ConnectionPolicyInstance>;
    /**
     * Streams ConnectionPolicyInstance records from the API.
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
     * @param { ConnectionPolicyListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ConnectionPolicyInstance, done: (err?: Error) => void) => void): void;
    each(params: ConnectionPolicyListInstanceEachOptions, callback?: (item: ConnectionPolicyInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ConnectionPolicyInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ConnectionPolicyPage) => any): Promise<ConnectionPolicyPage>;
    /**
     * Lists ConnectionPolicyInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ConnectionPolicyListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ConnectionPolicyInstance[]) => any): Promise<ConnectionPolicyInstance[]>;
    list(params: ConnectionPolicyListInstanceOptions, callback?: (error: Error | null, items: ConnectionPolicyInstance[]) => any): Promise<ConnectionPolicyInstance[]>;
    /**
     * Retrieve a single page of ConnectionPolicyInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ConnectionPolicyListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ConnectionPolicyPage) => any): Promise<ConnectionPolicyPage>;
    page(params: ConnectionPolicyListInstancePageOptions, callback?: (error: Error | null, items: ConnectionPolicyPage) => any): Promise<ConnectionPolicyPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ConnectionPolicyListInstance(version: V1): ConnectionPolicyListInstance;
export declare class ConnectionPolicyPage extends Page<V1, ConnectionPolicyPayload, ConnectionPolicyResource, ConnectionPolicyInstance> {
    /**
     * Initialize the ConnectionPolicyPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ConnectionPolicySolution);
    /**
     * Build an instance of ConnectionPolicyInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ConnectionPolicyResource): ConnectionPolicyInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
