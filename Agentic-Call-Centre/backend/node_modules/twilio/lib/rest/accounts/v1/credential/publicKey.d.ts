/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to update a PublicKeyInstance
 */
export interface PublicKeyContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
}
/**
 * Options to pass to create a PublicKeyInstance
 */
export interface PublicKeyListInstanceCreateOptions {
    /** A URL encoded representation of the public key. For example, `-----BEGIN PUBLIC KEY-----MIIBIjANB.pa9xQIDAQAB-----END PUBLIC KEY-----` */
    publicKey: string;
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** The SID of the Subaccount that this Credential should be associated with. Must be a valid Subaccount of the account issuing the request */
    accountSid?: string;
}
/**
 * Options to pass to each
 */
export interface PublicKeyListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: PublicKeyInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface PublicKeyListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface PublicKeyListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface PublicKeyContext {
    /**
     * Remove a PublicKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a PublicKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PublicKeyInstance
     */
    fetch(callback?: (error: Error | null, item?: PublicKeyInstance) => any): Promise<PublicKeyInstance>;
    /**
     * Update a PublicKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PublicKeyInstance
     */
    update(callback?: (error: Error | null, item?: PublicKeyInstance) => any): Promise<PublicKeyInstance>;
    /**
     * Update a PublicKeyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PublicKeyInstance
     */
    update(params: PublicKeyContextUpdateOptions, callback?: (error: Error | null, item?: PublicKeyInstance) => any): Promise<PublicKeyInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PublicKeyContextSolution {
    sid: string;
}
export declare class PublicKeyContextImpl implements PublicKeyContext {
    protected _version: V1;
    protected _solution: PublicKeyContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: PublicKeyInstance) => any): Promise<PublicKeyInstance>;
    update(params?: PublicKeyContextUpdateOptions | ((error: Error | null, item?: PublicKeyInstance) => any), callback?: (error: Error | null, item?: PublicKeyInstance) => any): Promise<PublicKeyInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PublicKeyContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PublicKeyPayload extends TwilioResponsePayload {
    credentials: PublicKeyResource[];
}
interface PublicKeyResource {
    sid: string;
    account_sid: string;
    friendly_name: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class PublicKeyInstance {
    protected _version: V1;
    protected _solution: PublicKeyContextSolution;
    protected _context?: PublicKeyContext;
    constructor(_version: V1, payload: PublicKeyResource, sid?: string);
    /**
     * The unique string that that we created to identify the PublicKey resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Credential that the PublicKey resource belongs to.
     */
    accountSid: string;
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
     * The URI for this resource, relative to `https://accounts.twilio.com`
     */
    url: string;
    private get _proxy();
    /**
     * Remove a PublicKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a PublicKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PublicKeyInstance
     */
    fetch(callback?: (error: Error | null, item?: PublicKeyInstance) => any): Promise<PublicKeyInstance>;
    /**
     * Update a PublicKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PublicKeyInstance
     */
    update(callback?: (error: Error | null, item?: PublicKeyInstance) => any): Promise<PublicKeyInstance>;
    /**
     * Update a PublicKeyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PublicKeyInstance
     */
    update(params: PublicKeyContextUpdateOptions, callback?: (error: Error | null, item?: PublicKeyInstance) => any): Promise<PublicKeyInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        friendlyName: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PublicKeySolution {
}
export interface PublicKeyListInstance {
    _version: V1;
    _solution: PublicKeySolution;
    _uri: string;
    (sid: string): PublicKeyContext;
    get(sid: string): PublicKeyContext;
    /**
     * Create a PublicKeyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PublicKeyInstance
     */
    create(params: PublicKeyListInstanceCreateOptions, callback?: (error: Error | null, item?: PublicKeyInstance) => any): Promise<PublicKeyInstance>;
    /**
     * Streams PublicKeyInstance records from the API.
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
     * @param { PublicKeyListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: PublicKeyInstance, done: (err?: Error) => void) => void): void;
    each(params: PublicKeyListInstanceEachOptions, callback?: (item: PublicKeyInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of PublicKeyInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: PublicKeyPage) => any): Promise<PublicKeyPage>;
    /**
     * Lists PublicKeyInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PublicKeyListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: PublicKeyInstance[]) => any): Promise<PublicKeyInstance[]>;
    list(params: PublicKeyListInstanceOptions, callback?: (error: Error | null, items: PublicKeyInstance[]) => any): Promise<PublicKeyInstance[]>;
    /**
     * Retrieve a single page of PublicKeyInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PublicKeyListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: PublicKeyPage) => any): Promise<PublicKeyPage>;
    page(params: PublicKeyListInstancePageOptions, callback?: (error: Error | null, items: PublicKeyPage) => any): Promise<PublicKeyPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PublicKeyListInstance(version: V1): PublicKeyListInstance;
export declare class PublicKeyPage extends Page<V1, PublicKeyPayload, PublicKeyResource, PublicKeyInstance> {
    /**
     * Initialize the PublicKeyPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: PublicKeySolution);
    /**
     * Build an instance of PublicKeyInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: PublicKeyResource): PublicKeyInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
