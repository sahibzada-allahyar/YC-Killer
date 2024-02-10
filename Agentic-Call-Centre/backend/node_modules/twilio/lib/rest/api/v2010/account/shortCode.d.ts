/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
/**
 * Options to pass to update a ShortCodeInstance
 */
export interface ShortCodeContextUpdateOptions {
    /** A descriptive string that you created to describe this resource. It can be up to 64 characters long. By default, the `FriendlyName` is the short code. */
    friendlyName?: string;
    /** The API version to use to start a new TwiML session. Can be: `2010-04-01` or `2008-08-01`. */
    apiVersion?: string;
    /** The URL we should call when receiving an incoming SMS message to this short code. */
    smsUrl?: string;
    /** The HTTP method we should use when calling the `sms_url`. Can be: `GET` or `POST`. */
    smsMethod?: string;
    /** The URL that we should call if an error occurs while retrieving or executing the TwiML from `sms_url`. */
    smsFallbackUrl?: string;
    /** The HTTP method that we should use to call the `sms_fallback_url`. Can be: `GET` or `POST`. */
    smsFallbackMethod?: string;
}
/**
 * Options to pass to each
 */
export interface ShortCodeListInstanceEachOptions {
    /** The string that identifies the ShortCode resources to read. */
    friendlyName?: string;
    /** Only show the ShortCode resources that match this pattern. You can specify partial numbers and use \'*\' as a wildcard for any digit. */
    shortCode?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ShortCodeInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ShortCodeListInstanceOptions {
    /** The string that identifies the ShortCode resources to read. */
    friendlyName?: string;
    /** Only show the ShortCode resources that match this pattern. You can specify partial numbers and use \'*\' as a wildcard for any digit. */
    shortCode?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ShortCodeListInstancePageOptions {
    /** The string that identifies the ShortCode resources to read. */
    friendlyName?: string;
    /** Only show the ShortCode resources that match this pattern. You can specify partial numbers and use \'*\' as a wildcard for any digit. */
    shortCode?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ShortCodeContext {
    /**
     * Fetch a ShortCodeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ShortCodeInstance
     */
    fetch(callback?: (error: Error | null, item?: ShortCodeInstance) => any): Promise<ShortCodeInstance>;
    /**
     * Update a ShortCodeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ShortCodeInstance
     */
    update(callback?: (error: Error | null, item?: ShortCodeInstance) => any): Promise<ShortCodeInstance>;
    /**
     * Update a ShortCodeInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ShortCodeInstance
     */
    update(params: ShortCodeContextUpdateOptions, callback?: (error: Error | null, item?: ShortCodeInstance) => any): Promise<ShortCodeInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ShortCodeContextSolution {
    accountSid: string;
    sid: string;
}
export declare class ShortCodeContextImpl implements ShortCodeContext {
    protected _version: V2010;
    protected _solution: ShortCodeContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: ShortCodeInstance) => any): Promise<ShortCodeInstance>;
    update(params?: ShortCodeContextUpdateOptions | ((error: Error | null, item?: ShortCodeInstance) => any), callback?: (error: Error | null, item?: ShortCodeInstance) => any): Promise<ShortCodeInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ShortCodeContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ShortCodePayload extends TwilioResponsePayload {
    short_codes: ShortCodeResource[];
}
interface ShortCodeResource {
    account_sid: string;
    api_version: string;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    short_code: string;
    sid: string;
    sms_fallback_method: string;
    sms_fallback_url: string;
    sms_method: string;
    sms_url: string;
    uri: string;
}
export declare class ShortCodeInstance {
    protected _version: V2010;
    protected _solution: ShortCodeContextSolution;
    protected _context?: ShortCodeContext;
    constructor(_version: V2010, payload: ShortCodeResource, accountSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created this ShortCode resource.
     */
    accountSid: string;
    /**
     * The API version used to start a new TwiML session when an SMS message is sent to this short code.
     */
    apiVersion: string;
    /**
     * The date and time in GMT that this resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that this resource was last updated, specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * A string that you assigned to describe this resource. By default, the `FriendlyName` is the short code.
     */
    friendlyName: string;
    /**
     * The short code. e.g., 894546.
     */
    shortCode: string;
    /**
     * The unique string that that we created to identify this ShortCode resource.
     */
    sid: string;
    /**
     * The HTTP method we use to call the `sms_fallback_url`. Can be: `GET` or `POST`.
     */
    smsFallbackMethod: string;
    /**
     * The URL that we call if an error occurs while retrieving or executing the TwiML from `sms_url`.
     */
    smsFallbackUrl: string;
    /**
     * The HTTP method we use to call the `sms_url`. Can be: `GET` or `POST`.
     */
    smsMethod: string;
    /**
     * The URL we call when receiving an incoming SMS message to this short code.
     */
    smsUrl: string;
    /**
     * The URI of this resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Fetch a ShortCodeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ShortCodeInstance
     */
    fetch(callback?: (error: Error | null, item?: ShortCodeInstance) => any): Promise<ShortCodeInstance>;
    /**
     * Update a ShortCodeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ShortCodeInstance
     */
    update(callback?: (error: Error | null, item?: ShortCodeInstance) => any): Promise<ShortCodeInstance>;
    /**
     * Update a ShortCodeInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ShortCodeInstance
     */
    update(params: ShortCodeContextUpdateOptions, callback?: (error: Error | null, item?: ShortCodeInstance) => any): Promise<ShortCodeInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        apiVersion: string;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        shortCode: string;
        sid: string;
        smsFallbackMethod: string;
        smsFallbackUrl: string;
        smsMethod: string;
        smsUrl: string;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ShortCodeSolution {
    accountSid: string;
}
export interface ShortCodeListInstance {
    _version: V2010;
    _solution: ShortCodeSolution;
    _uri: string;
    (sid: string): ShortCodeContext;
    get(sid: string): ShortCodeContext;
    /**
     * Streams ShortCodeInstance records from the API.
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
     * @param { ShortCodeListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ShortCodeInstance, done: (err?: Error) => void) => void): void;
    each(params: ShortCodeListInstanceEachOptions, callback?: (item: ShortCodeInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ShortCodeInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ShortCodePage) => any): Promise<ShortCodePage>;
    /**
     * Lists ShortCodeInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ShortCodeListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ShortCodeInstance[]) => any): Promise<ShortCodeInstance[]>;
    list(params: ShortCodeListInstanceOptions, callback?: (error: Error | null, items: ShortCodeInstance[]) => any): Promise<ShortCodeInstance[]>;
    /**
     * Retrieve a single page of ShortCodeInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ShortCodeListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ShortCodePage) => any): Promise<ShortCodePage>;
    page(params: ShortCodeListInstancePageOptions, callback?: (error: Error | null, items: ShortCodePage) => any): Promise<ShortCodePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ShortCodeListInstance(version: V2010, accountSid: string): ShortCodeListInstance;
export declare class ShortCodePage extends Page<V2010, ShortCodePayload, ShortCodeResource, ShortCodeInstance> {
    /**
     * Initialize the ShortCodePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: ShortCodeSolution);
    /**
     * Build an instance of ShortCodeInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ShortCodeResource): ShortCodeInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
