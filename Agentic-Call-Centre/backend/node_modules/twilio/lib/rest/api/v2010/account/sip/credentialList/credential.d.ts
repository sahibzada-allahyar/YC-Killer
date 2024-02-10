/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../../base/Page";
import Response from "../../../../../../http/response";
import V2010 from "../../../../V2010";
/**
 * Options to pass to update a CredentialInstance
 */
export interface CredentialContextUpdateOptions {
    /** The password that the username will use when authenticating SIP requests. The password must be a minimum of 12 characters, contain at least 1 digit, and have mixed case. (eg `IWasAtSignal2018`) */
    password?: string;
}
/**
 * Options to pass to create a CredentialInstance
 */
export interface CredentialListInstanceCreateOptions {
    /** The username that will be passed when authenticating SIP requests. The username should be sent in response to Twilio\\\'s challenge of the initial INVITE. It can be up to 32 characters long. */
    username: string;
    /** The password that the username will use when authenticating SIP requests. The password must be a minimum of 12 characters, contain at least 1 digit, and have mixed case. (eg `IWasAtSignal2018`) */
    password: string;
}
/**
 * Options to pass to each
 */
export interface CredentialListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: CredentialInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface CredentialListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CredentialListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CredentialContext {
    /**
     * Remove a CredentialInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CredentialInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CredentialInstance
     */
    fetch(callback?: (error: Error | null, item?: CredentialInstance) => any): Promise<CredentialInstance>;
    /**
     * Update a CredentialInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CredentialInstance
     */
    update(callback?: (error: Error | null, item?: CredentialInstance) => any): Promise<CredentialInstance>;
    /**
     * Update a CredentialInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CredentialInstance
     */
    update(params: CredentialContextUpdateOptions, callback?: (error: Error | null, item?: CredentialInstance) => any): Promise<CredentialInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface CredentialContextSolution {
    accountSid: string;
    credentialListSid: string;
    sid: string;
}
export declare class CredentialContextImpl implements CredentialContext {
    protected _version: V2010;
    protected _solution: CredentialContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, credentialListSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: CredentialInstance) => any): Promise<CredentialInstance>;
    update(params?: CredentialContextUpdateOptions | ((error: Error | null, item?: CredentialInstance) => any), callback?: (error: Error | null, item?: CredentialInstance) => any): Promise<CredentialInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): CredentialContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface CredentialPayload extends TwilioResponsePayload {
    credentials: CredentialResource[];
}
interface CredentialResource {
    sid: string;
    account_sid: string;
    credential_list_sid: string;
    username: string;
    date_created: Date;
    date_updated: Date;
    uri: string;
}
export declare class CredentialInstance {
    protected _version: V2010;
    protected _solution: CredentialContextSolution;
    protected _context?: CredentialContext;
    constructor(_version: V2010, payload: CredentialResource, accountSid: string, credentialListSid: string, sid?: string);
    /**
     * A 34 character string that uniquely identifies this resource.
     */
    sid: string;
    /**
     * The unique id of the Account that is responsible for this resource.
     */
    accountSid: string;
    /**
     * The unique id that identifies the credential list that includes this credential.
     */
    credentialListSid: string;
    /**
     * The username for this credential.
     */
    username: string;
    /**
     * The date that this resource was created, given as GMT in [RFC 2822](https://www.php.net/manual/en/class.datetime.php#datetime.constants.rfc2822) format.
     */
    dateCreated: Date;
    /**
     * The date that this resource was last updated, given as GMT in [RFC 2822](https://www.php.net/manual/en/class.datetime.php#datetime.constants.rfc2822) format.
     */
    dateUpdated: Date;
    /**
     * The URI for this resource, relative to `https://api.twilio.com`
     */
    uri: string;
    private get _proxy();
    /**
     * Remove a CredentialInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CredentialInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CredentialInstance
     */
    fetch(callback?: (error: Error | null, item?: CredentialInstance) => any): Promise<CredentialInstance>;
    /**
     * Update a CredentialInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CredentialInstance
     */
    update(callback?: (error: Error | null, item?: CredentialInstance) => any): Promise<CredentialInstance>;
    /**
     * Update a CredentialInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CredentialInstance
     */
    update(params: CredentialContextUpdateOptions, callback?: (error: Error | null, item?: CredentialInstance) => any): Promise<CredentialInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        credentialListSid: string;
        username: string;
        dateCreated: Date;
        dateUpdated: Date;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CredentialSolution {
    accountSid: string;
    credentialListSid: string;
}
export interface CredentialListInstance {
    _version: V2010;
    _solution: CredentialSolution;
    _uri: string;
    (sid: string): CredentialContext;
    get(sid: string): CredentialContext;
    /**
     * Create a CredentialInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CredentialInstance
     */
    create(params: CredentialListInstanceCreateOptions, callback?: (error: Error | null, item?: CredentialInstance) => any): Promise<CredentialInstance>;
    /**
     * Streams CredentialInstance records from the API.
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
     * @param { CredentialListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: CredentialInstance, done: (err?: Error) => void) => void): void;
    each(params: CredentialListInstanceEachOptions, callback?: (item: CredentialInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of CredentialInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: CredentialPage) => any): Promise<CredentialPage>;
    /**
     * Lists CredentialInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CredentialListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: CredentialInstance[]) => any): Promise<CredentialInstance[]>;
    list(params: CredentialListInstanceOptions, callback?: (error: Error | null, items: CredentialInstance[]) => any): Promise<CredentialInstance[]>;
    /**
     * Retrieve a single page of CredentialInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CredentialListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: CredentialPage) => any): Promise<CredentialPage>;
    page(params: CredentialListInstancePageOptions, callback?: (error: Error | null, items: CredentialPage) => any): Promise<CredentialPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CredentialListInstance(version: V2010, accountSid: string, credentialListSid: string): CredentialListInstance;
export declare class CredentialPage extends Page<V2010, CredentialPayload, CredentialResource, CredentialInstance> {
    /**
     * Initialize the CredentialPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: CredentialSolution);
    /**
     * Build an instance of CredentialInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CredentialResource): CredentialInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
