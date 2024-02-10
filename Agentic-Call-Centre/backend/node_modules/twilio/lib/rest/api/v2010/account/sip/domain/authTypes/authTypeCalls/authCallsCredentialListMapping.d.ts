/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../../../../base/Page";
import Response from "../../../../../../../../http/response";
import V2010 from "../../../../../../V2010";
/**
 * Options to pass to create a AuthCallsCredentialListMappingInstance
 */
export interface AuthCallsCredentialListMappingListInstanceCreateOptions {
    /** The SID of the CredentialList resource to map to the SIP domain. */
    credentialListSid: string;
}
/**
 * Options to pass to each
 */
export interface AuthCallsCredentialListMappingListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AuthCallsCredentialListMappingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AuthCallsCredentialListMappingListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AuthCallsCredentialListMappingListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AuthCallsCredentialListMappingContext {
    /**
     * Remove a AuthCallsCredentialListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AuthCallsCredentialListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthCallsCredentialListMappingInstance
     */
    fetch(callback?: (error: Error | null, item?: AuthCallsCredentialListMappingInstance) => any): Promise<AuthCallsCredentialListMappingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AuthCallsCredentialListMappingContextSolution {
    accountSid: string;
    domainSid: string;
    sid: string;
}
export declare class AuthCallsCredentialListMappingContextImpl implements AuthCallsCredentialListMappingContext {
    protected _version: V2010;
    protected _solution: AuthCallsCredentialListMappingContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, domainSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AuthCallsCredentialListMappingInstance) => any): Promise<AuthCallsCredentialListMappingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AuthCallsCredentialListMappingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AuthCallsCredentialListMappingPayload extends TwilioResponsePayload {
    contents: AuthCallsCredentialListMappingResource[];
}
interface AuthCallsCredentialListMappingResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    sid: string;
}
export declare class AuthCallsCredentialListMappingInstance {
    protected _version: V2010;
    protected _solution: AuthCallsCredentialListMappingContextSolution;
    protected _context?: AuthCallsCredentialListMappingContext;
    constructor(_version: V2010, payload: AuthCallsCredentialListMappingResource, accountSid: string, domainSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the CredentialListMapping resource.
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
     * The unique string that that we created to identify the CredentialListMapping resource.
     */
    sid: string;
    private get _proxy();
    /**
     * Remove a AuthCallsCredentialListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AuthCallsCredentialListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthCallsCredentialListMappingInstance
     */
    fetch(callback?: (error: Error | null, item?: AuthCallsCredentialListMappingInstance) => any): Promise<AuthCallsCredentialListMappingInstance>;
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
export interface AuthCallsCredentialListMappingSolution {
    accountSid: string;
    domainSid: string;
}
export interface AuthCallsCredentialListMappingListInstance {
    _version: V2010;
    _solution: AuthCallsCredentialListMappingSolution;
    _uri: string;
    (sid: string): AuthCallsCredentialListMappingContext;
    get(sid: string): AuthCallsCredentialListMappingContext;
    /**
     * Create a AuthCallsCredentialListMappingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthCallsCredentialListMappingInstance
     */
    create(params: AuthCallsCredentialListMappingListInstanceCreateOptions, callback?: (error: Error | null, item?: AuthCallsCredentialListMappingInstance) => any): Promise<AuthCallsCredentialListMappingInstance>;
    /**
     * Streams AuthCallsCredentialListMappingInstance records from the API.
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
     * @param { AuthCallsCredentialListMappingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AuthCallsCredentialListMappingInstance, done: (err?: Error) => void) => void): void;
    each(params: AuthCallsCredentialListMappingListInstanceEachOptions, callback?: (item: AuthCallsCredentialListMappingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AuthCallsCredentialListMappingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AuthCallsCredentialListMappingPage) => any): Promise<AuthCallsCredentialListMappingPage>;
    /**
     * Lists AuthCallsCredentialListMappingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AuthCallsCredentialListMappingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AuthCallsCredentialListMappingInstance[]) => any): Promise<AuthCallsCredentialListMappingInstance[]>;
    list(params: AuthCallsCredentialListMappingListInstanceOptions, callback?: (error: Error | null, items: AuthCallsCredentialListMappingInstance[]) => any): Promise<AuthCallsCredentialListMappingInstance[]>;
    /**
     * Retrieve a single page of AuthCallsCredentialListMappingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AuthCallsCredentialListMappingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AuthCallsCredentialListMappingPage) => any): Promise<AuthCallsCredentialListMappingPage>;
    page(params: AuthCallsCredentialListMappingListInstancePageOptions, callback?: (error: Error | null, items: AuthCallsCredentialListMappingPage) => any): Promise<AuthCallsCredentialListMappingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AuthCallsCredentialListMappingListInstance(version: V2010, accountSid: string, domainSid: string): AuthCallsCredentialListMappingListInstance;
export declare class AuthCallsCredentialListMappingPage extends Page<V2010, AuthCallsCredentialListMappingPayload, AuthCallsCredentialListMappingResource, AuthCallsCredentialListMappingInstance> {
    /**
     * Initialize the AuthCallsCredentialListMappingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: AuthCallsCredentialListMappingSolution);
    /**
     * Build an instance of AuthCallsCredentialListMappingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AuthCallsCredentialListMappingResource): AuthCallsCredentialListMappingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
