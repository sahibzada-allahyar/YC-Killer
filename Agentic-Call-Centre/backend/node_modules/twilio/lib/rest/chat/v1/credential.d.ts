/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type CredentialPushService = "gcm" | "apn" | "fcm";
/**
 * Options to pass to update a CredentialInstance
 */
export interface CredentialContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** [APN only] The URL encoded representation of the certificate. For example,  `-----BEGIN CERTIFICATE----- MIIFnTCCBIWgAwIBAgIIAjy9H849+E8wDQYJKoZIhvcNAQEFBQAwgZYxCzAJBgNV.....A== -----END CERTIFICATE-----` */
    certificate?: string;
    /** [APN only] The URL encoded representation of the private key. For example, `-----BEGIN RSA PRIVATE KEY----- MIIEpQIBAAKCAQEAuyf/lNrH9ck8DmNyo3fGgvCI1l9s+cmBY3WIz+cUDqmxiieR. -----END RSA PRIVATE KEY-----` */
    privateKey?: string;
    /** [APN only] Whether to send the credential to sandbox APNs. Can be `true` to send to sandbox APNs or `false` to send to production. */
    sandbox?: boolean;
    /** [GCM only] The API key for the project that was obtained from the Google Developer console for your GCM Service application credential. */
    apiKey?: string;
    /** [FCM only] The **Server key** of your project from the Firebase console, found under Settings / Cloud messaging. */
    secret?: string;
}
/**
 * Options to pass to create a CredentialInstance
 */
export interface CredentialListInstanceCreateOptions {
    /**  */
    type: CredentialPushService;
    /** A descriptive string that you create to describe the new resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** [APN only] The URL encoded representation of the certificate. For example,  `-----BEGIN CERTIFICATE----- MIIFnTCCBIWgAwIBAgIIAjy9H849+E8wDQYJKoZIhvcNAQEFBQAwgZYxCzAJBgNV.....A== -----END CERTIFICATE-----` */
    certificate?: string;
    /** [APN only] The URL encoded representation of the private key. For example, `-----BEGIN RSA PRIVATE KEY----- MIIEpQIBAAKCAQEAuyf/lNrH9ck8DmNyo3fGgvCI1l9s+cmBY3WIz+cUDqmxiieR. -----END RSA PRIVATE KEY-----` */
    privateKey?: string;
    /** [APN only] Whether to send the credential to sandbox APNs. Can be `true` to send to sandbox APNs or `false` to send to production. */
    sandbox?: boolean;
    /** [GCM only] The API key for the project that was obtained from the Google Developer console for your GCM Service application credential. */
    apiKey?: string;
    /** [FCM only] The **Server key** of your project from the Firebase console, found under Settings / Cloud messaging. */
    secret?: string;
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
    sid: string;
}
export declare class CredentialContextImpl implements CredentialContext {
    protected _version: V1;
    protected _solution: CredentialContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
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
    friendly_name: string;
    type: CredentialPushService;
    sandbox: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class CredentialInstance {
    protected _version: V1;
    protected _solution: CredentialContextSolution;
    protected _context?: CredentialContext;
    constructor(_version: V1, payload: CredentialResource, sid?: string);
    /**
     * The unique string that we created to identify the Credential resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/api/rest/account) that created the Credential resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    type: CredentialPushService;
    /**
     * [APN only] Whether to send the credential to sandbox APNs. Can be `true` to send to sandbox APNs or `false` to send to production.
     */
    sandbox: string;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](http://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](http://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Credential resource.
     */
    url: string;
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
        friendlyName: string;
        type: CredentialPushService;
        sandbox: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CredentialSolution {
}
export interface CredentialListInstance {
    _version: V1;
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
export declare function CredentialListInstance(version: V1): CredentialListInstance;
export declare class CredentialPage extends Page<V1, CredentialPayload, CredentialResource, CredentialInstance> {
    /**
     * Initialize the CredentialPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: CredentialSolution);
    /**
     * Build an instance of CredentialInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CredentialResource): CredentialInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
