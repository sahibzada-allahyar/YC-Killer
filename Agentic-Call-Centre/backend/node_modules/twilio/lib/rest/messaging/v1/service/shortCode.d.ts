/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to create a ShortCodeInstance
 */
export interface ShortCodeListInstanceCreateOptions {
    /** The SID of the ShortCode resource being added to the Service. */
    shortCodeSid: string;
}
/**
 * Options to pass to each
 */
export interface ShortCodeListInstanceEachOptions {
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
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ShortCodeListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ShortCodeContext {
    /**
     * Remove a ShortCodeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ShortCodeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ShortCodeInstance
     */
    fetch(callback?: (error: Error | null, item?: ShortCodeInstance) => any): Promise<ShortCodeInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ShortCodeContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class ShortCodeContextImpl implements ShortCodeContext {
    protected _version: V1;
    protected _solution: ShortCodeContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ShortCodeInstance) => any): Promise<ShortCodeInstance>;
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
    sid: string;
    account_sid: string;
    service_sid: string;
    date_created: Date;
    date_updated: Date;
    short_code: string;
    country_code: string;
    capabilities: Array<string>;
    url: string;
}
export declare class ShortCodeInstance {
    protected _version: V1;
    protected _solution: ShortCodeContextSolution;
    protected _context?: ShortCodeContext;
    constructor(_version: V1, payload: ShortCodeResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the ShortCode resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the ShortCode resource.
     */
    accountSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/chat/rest/service-resource) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The [E.164](https://www.twilio.com/docs/glossary/what-e164) format of the short code.
     */
    shortCode: string;
    /**
     * The 2-character [ISO Country Code](https://www.iso.org/iso-3166-country-codes.html) of the number.
     */
    countryCode: string;
    /**
     * An array of values that describe whether the number can receive calls or messages. Can be: `SMS` and `MMS`.
     */
    capabilities: Array<string>;
    /**
     * The absolute URL of the ShortCode resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a ShortCodeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ShortCodeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ShortCodeInstance
     */
    fetch(callback?: (error: Error | null, item?: ShortCodeInstance) => any): Promise<ShortCodeInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        shortCode: string;
        countryCode: string;
        capabilities: string[];
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ShortCodeSolution {
    serviceSid: string;
}
export interface ShortCodeListInstance {
    _version: V1;
    _solution: ShortCodeSolution;
    _uri: string;
    (sid: string): ShortCodeContext;
    get(sid: string): ShortCodeContext;
    /**
     * Create a ShortCodeInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ShortCodeInstance
     */
    create(params: ShortCodeListInstanceCreateOptions, callback?: (error: Error | null, item?: ShortCodeInstance) => any): Promise<ShortCodeInstance>;
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
export declare function ShortCodeListInstance(version: V1, serviceSid: string): ShortCodeListInstance;
export declare class ShortCodePage extends Page<V1, ShortCodePayload, ShortCodeResource, ShortCodeInstance> {
    /**
     * Initialize the ShortCodePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ShortCodeSolution);
    /**
     * Build an instance of ShortCodeInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ShortCodeResource): ShortCodeInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
