/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../../../../base/Page";
import Response from "../../../../../../../../http/response";
import V2010 from "../../../../../../V2010";
/**
 * Options to pass to create a AuthCallsIpAccessControlListMappingInstance
 */
export interface AuthCallsIpAccessControlListMappingListInstanceCreateOptions {
    /** The SID of the IpAccessControlList resource to map to the SIP domain. */
    ipAccessControlListSid: string;
}
/**
 * Options to pass to each
 */
export interface AuthCallsIpAccessControlListMappingListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AuthCallsIpAccessControlListMappingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AuthCallsIpAccessControlListMappingListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AuthCallsIpAccessControlListMappingListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AuthCallsIpAccessControlListMappingContext {
    /**
     * Remove a AuthCallsIpAccessControlListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AuthCallsIpAccessControlListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthCallsIpAccessControlListMappingInstance
     */
    fetch(callback?: (error: Error | null, item?: AuthCallsIpAccessControlListMappingInstance) => any): Promise<AuthCallsIpAccessControlListMappingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AuthCallsIpAccessControlListMappingContextSolution {
    accountSid: string;
    domainSid: string;
    sid: string;
}
export declare class AuthCallsIpAccessControlListMappingContextImpl implements AuthCallsIpAccessControlListMappingContext {
    protected _version: V2010;
    protected _solution: AuthCallsIpAccessControlListMappingContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, domainSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AuthCallsIpAccessControlListMappingInstance) => any): Promise<AuthCallsIpAccessControlListMappingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AuthCallsIpAccessControlListMappingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AuthCallsIpAccessControlListMappingPayload extends TwilioResponsePayload {
    contents: AuthCallsIpAccessControlListMappingResource[];
}
interface AuthCallsIpAccessControlListMappingResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    sid: string;
}
export declare class AuthCallsIpAccessControlListMappingInstance {
    protected _version: V2010;
    protected _solution: AuthCallsIpAccessControlListMappingContextSolution;
    protected _context?: AuthCallsIpAccessControlListMappingContext;
    constructor(_version: V2010, payload: AuthCallsIpAccessControlListMappingResource, accountSid: string, domainSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the IpAccessControlListMapping resource.
     */
    accountSid: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The unique string that that we created to identify the IpAccessControlListMapping resource.
     */
    sid: string;
    private get _proxy();
    /**
     * Remove a AuthCallsIpAccessControlListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AuthCallsIpAccessControlListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthCallsIpAccessControlListMappingInstance
     */
    fetch(callback?: (error: Error | null, item?: AuthCallsIpAccessControlListMappingInstance) => any): Promise<AuthCallsIpAccessControlListMappingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        sid: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AuthCallsIpAccessControlListMappingSolution {
    accountSid: string;
    domainSid: string;
}
export interface AuthCallsIpAccessControlListMappingListInstance {
    _version: V2010;
    _solution: AuthCallsIpAccessControlListMappingSolution;
    _uri: string;
    (sid: string): AuthCallsIpAccessControlListMappingContext;
    get(sid: string): AuthCallsIpAccessControlListMappingContext;
    /**
     * Create a AuthCallsIpAccessControlListMappingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthCallsIpAccessControlListMappingInstance
     */
    create(params: AuthCallsIpAccessControlListMappingListInstanceCreateOptions, callback?: (error: Error | null, item?: AuthCallsIpAccessControlListMappingInstance) => any): Promise<AuthCallsIpAccessControlListMappingInstance>;
    /**
     * Streams AuthCallsIpAccessControlListMappingInstance records from the API.
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
     * @param { AuthCallsIpAccessControlListMappingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AuthCallsIpAccessControlListMappingInstance, done: (err?: Error) => void) => void): void;
    each(params: AuthCallsIpAccessControlListMappingListInstanceEachOptions, callback?: (item: AuthCallsIpAccessControlListMappingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AuthCallsIpAccessControlListMappingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AuthCallsIpAccessControlListMappingPage) => any): Promise<AuthCallsIpAccessControlListMappingPage>;
    /**
     * Lists AuthCallsIpAccessControlListMappingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AuthCallsIpAccessControlListMappingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AuthCallsIpAccessControlListMappingInstance[]) => any): Promise<AuthCallsIpAccessControlListMappingInstance[]>;
    list(params: AuthCallsIpAccessControlListMappingListInstanceOptions, callback?: (error: Error | null, items: AuthCallsIpAccessControlListMappingInstance[]) => any): Promise<AuthCallsIpAccessControlListMappingInstance[]>;
    /**
     * Retrieve a single page of AuthCallsIpAccessControlListMappingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AuthCallsIpAccessControlListMappingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AuthCallsIpAccessControlListMappingPage) => any): Promise<AuthCallsIpAccessControlListMappingPage>;
    page(params: AuthCallsIpAccessControlListMappingListInstancePageOptions, callback?: (error: Error | null, items: AuthCallsIpAccessControlListMappingPage) => any): Promise<AuthCallsIpAccessControlListMappingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AuthCallsIpAccessControlListMappingListInstance(version: V2010, accountSid: string, domainSid: string): AuthCallsIpAccessControlListMappingListInstance;
export declare class AuthCallsIpAccessControlListMappingPage extends Page<V2010, AuthCallsIpAccessControlListMappingPayload, AuthCallsIpAccessControlListMappingResource, AuthCallsIpAccessControlListMappingInstance> {
    /**
     * Initialize the AuthCallsIpAccessControlListMappingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: AuthCallsIpAccessControlListMappingSolution);
    /**
     * Build an instance of AuthCallsIpAccessControlListMappingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AuthCallsIpAccessControlListMappingResource): AuthCallsIpAccessControlListMappingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
