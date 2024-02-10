/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../../../../base/Page";
import Response from "../../../../../../../../http/response";
import V2010 from "../../../../../../V2010";
/**
 * Options to pass to create a AuthRegistrationsCredentialListMappingInstance
 */
export interface AuthRegistrationsCredentialListMappingListInstanceCreateOptions {
    /** The SID of the CredentialList resource to map to the SIP domain. */
    credentialListSid: string;
}
/**
 * Options to pass to each
 */
export interface AuthRegistrationsCredentialListMappingListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AuthRegistrationsCredentialListMappingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AuthRegistrationsCredentialListMappingListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AuthRegistrationsCredentialListMappingListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AuthRegistrationsCredentialListMappingContext {
    /**
     * Remove a AuthRegistrationsCredentialListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AuthRegistrationsCredentialListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthRegistrationsCredentialListMappingInstance
     */
    fetch(callback?: (error: Error | null, item?: AuthRegistrationsCredentialListMappingInstance) => any): Promise<AuthRegistrationsCredentialListMappingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AuthRegistrationsCredentialListMappingContextSolution {
    accountSid: string;
    domainSid: string;
    sid: string;
}
export declare class AuthRegistrationsCredentialListMappingContextImpl implements AuthRegistrationsCredentialListMappingContext {
    protected _version: V2010;
    protected _solution: AuthRegistrationsCredentialListMappingContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, domainSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AuthRegistrationsCredentialListMappingInstance) => any): Promise<AuthRegistrationsCredentialListMappingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AuthRegistrationsCredentialListMappingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AuthRegistrationsCredentialListMappingPayload extends TwilioResponsePayload {
    contents: AuthRegistrationsCredentialListMappingResource[];
}
interface AuthRegistrationsCredentialListMappingResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    sid: string;
}
export declare class AuthRegistrationsCredentialListMappingInstance {
    protected _version: V2010;
    protected _solution: AuthRegistrationsCredentialListMappingContextSolution;
    protected _context?: AuthRegistrationsCredentialListMappingContext;
    constructor(_version: V2010, payload: AuthRegistrationsCredentialListMappingResource, accountSid: string, domainSid: string, sid?: string);
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
     * Remove a AuthRegistrationsCredentialListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AuthRegistrationsCredentialListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthRegistrationsCredentialListMappingInstance
     */
    fetch(callback?: (error: Error | null, item?: AuthRegistrationsCredentialListMappingInstance) => any): Promise<AuthRegistrationsCredentialListMappingInstance>;
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
export interface AuthRegistrationsCredentialListMappingSolution {
    accountSid: string;
    domainSid: string;
}
export interface AuthRegistrationsCredentialListMappingListInstance {
    _version: V2010;
    _solution: AuthRegistrationsCredentialListMappingSolution;
    _uri: string;
    (sid: string): AuthRegistrationsCredentialListMappingContext;
    get(sid: string): AuthRegistrationsCredentialListMappingContext;
    /**
     * Create a AuthRegistrationsCredentialListMappingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthRegistrationsCredentialListMappingInstance
     */
    create(params: AuthRegistrationsCredentialListMappingListInstanceCreateOptions, callback?: (error: Error | null, item?: AuthRegistrationsCredentialListMappingInstance) => any): Promise<AuthRegistrationsCredentialListMappingInstance>;
    /**
     * Streams AuthRegistrationsCredentialListMappingInstance records from the API.
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
     * @param { AuthRegistrationsCredentialListMappingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AuthRegistrationsCredentialListMappingInstance, done: (err?: Error) => void) => void): void;
    each(params: AuthRegistrationsCredentialListMappingListInstanceEachOptions, callback?: (item: AuthRegistrationsCredentialListMappingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AuthRegistrationsCredentialListMappingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AuthRegistrationsCredentialListMappingPage) => any): Promise<AuthRegistrationsCredentialListMappingPage>;
    /**
     * Lists AuthRegistrationsCredentialListMappingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AuthRegistrationsCredentialListMappingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AuthRegistrationsCredentialListMappingInstance[]) => any): Promise<AuthRegistrationsCredentialListMappingInstance[]>;
    list(params: AuthRegistrationsCredentialListMappingListInstanceOptions, callback?: (error: Error | null, items: AuthRegistrationsCredentialListMappingInstance[]) => any): Promise<AuthRegistrationsCredentialListMappingInstance[]>;
    /**
     * Retrieve a single page of AuthRegistrationsCredentialListMappingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AuthRegistrationsCredentialListMappingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AuthRegistrationsCredentialListMappingPage) => any): Promise<AuthRegistrationsCredentialListMappingPage>;
    page(params: AuthRegistrationsCredentialListMappingListInstancePageOptions, callback?: (error: Error | null, items: AuthRegistrationsCredentialListMappingPage) => any): Promise<AuthRegistrationsCredentialListMappingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AuthRegistrationsCredentialListMappingListInstance(version: V2010, accountSid: string, domainSid: string): AuthRegistrationsCredentialListMappingListInstance;
export declare class AuthRegistrationsCredentialListMappingPage extends Page<V2010, AuthRegistrationsCredentialListMappingPayload, AuthRegistrationsCredentialListMappingResource, AuthRegistrationsCredentialListMappingInstance> {
    /**
     * Initialize the AuthRegistrationsCredentialListMappingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: AuthRegistrationsCredentialListMappingSolution);
    /**
     * Build an instance of AuthRegistrationsCredentialListMappingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AuthRegistrationsCredentialListMappingResource): AuthRegistrationsCredentialListMappingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
