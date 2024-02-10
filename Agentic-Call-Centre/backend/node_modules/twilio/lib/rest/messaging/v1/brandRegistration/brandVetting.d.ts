/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type BrandVettingVettingProvider = "campaign-verify";
/**
 * Options to pass to create a BrandVettingInstance
 */
export interface BrandVettingListInstanceCreateOptions {
    /**  */
    vettingProvider: BrandVettingVettingProvider;
    /** The unique ID of the vetting */
    vettingId?: string;
}
/**
 * Options to pass to each
 */
export interface BrandVettingListInstanceEachOptions {
    /** The third-party provider of the vettings to read */
    vettingProvider?: BrandVettingVettingProvider;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: BrandVettingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface BrandVettingListInstanceOptions {
    /** The third-party provider of the vettings to read */
    vettingProvider?: BrandVettingVettingProvider;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface BrandVettingListInstancePageOptions {
    /** The third-party provider of the vettings to read */
    vettingProvider?: BrandVettingVettingProvider;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface BrandVettingContext {
    /**
     * Fetch a BrandVettingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BrandVettingInstance
     */
    fetch(callback?: (error: Error | null, item?: BrandVettingInstance) => any): Promise<BrandVettingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface BrandVettingContextSolution {
    brandSid: string;
    brandVettingSid: string;
}
export declare class BrandVettingContextImpl implements BrandVettingContext {
    protected _version: V1;
    protected _solution: BrandVettingContextSolution;
    protected _uri: string;
    constructor(_version: V1, brandSid: string, brandVettingSid: string);
    fetch(callback?: (error: Error | null, item?: BrandVettingInstance) => any): Promise<BrandVettingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): BrandVettingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface BrandVettingPayload extends TwilioResponsePayload {
    data: BrandVettingResource[];
}
interface BrandVettingResource {
    account_sid: string;
    brand_sid: string;
    brand_vetting_sid: string;
    date_updated: Date;
    date_created: Date;
    vetting_id: string;
    vetting_class: string;
    vetting_status: string;
    vetting_provider: BrandVettingVettingProvider;
    url: string;
}
export declare class BrandVettingInstance {
    protected _version: V1;
    protected _solution: BrandVettingContextSolution;
    protected _context?: BrandVettingContext;
    constructor(_version: V1, payload: BrandVettingResource, brandSid: string, brandVettingSid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the vetting record.
     */
    accountSid: string;
    /**
     * The unique string to identify Brand Registration.
     */
    brandSid: string;
    /**
     * The Twilio SID of the third-party vetting record.
     */
    brandVettingSid: string;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The unique identifier of the vetting from the third-party provider.
     */
    vettingId: string;
    /**
     * The type of vetting that has been conducted. One of “STANDARD” (Aegis) or “POLITICAL” (Campaign Verify).
     */
    vettingClass: string;
    /**
     * The status of the import vetting attempt. One of “PENDING,” “SUCCESS,” or “FAILED”.
     */
    vettingStatus: string;
    vettingProvider: BrandVettingVettingProvider;
    /**
     * The absolute URL of the Brand Vetting resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a BrandVettingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BrandVettingInstance
     */
    fetch(callback?: (error: Error | null, item?: BrandVettingInstance) => any): Promise<BrandVettingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        brandSid: string;
        brandVettingSid: string;
        dateUpdated: Date;
        dateCreated: Date;
        vettingId: string;
        vettingClass: string;
        vettingStatus: string;
        vettingProvider: "campaign-verify";
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface BrandVettingSolution {
    brandSid: string;
}
export interface BrandVettingListInstance {
    _version: V1;
    _solution: BrandVettingSolution;
    _uri: string;
    (brandVettingSid: string): BrandVettingContext;
    get(brandVettingSid: string): BrandVettingContext;
    /**
     * Create a BrandVettingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BrandVettingInstance
     */
    create(params: BrandVettingListInstanceCreateOptions, callback?: (error: Error | null, item?: BrandVettingInstance) => any): Promise<BrandVettingInstance>;
    /**
     * Streams BrandVettingInstance records from the API.
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
     * @param { BrandVettingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: BrandVettingInstance, done: (err?: Error) => void) => void): void;
    each(params: BrandVettingListInstanceEachOptions, callback?: (item: BrandVettingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of BrandVettingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: BrandVettingPage) => any): Promise<BrandVettingPage>;
    /**
     * Lists BrandVettingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BrandVettingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: BrandVettingInstance[]) => any): Promise<BrandVettingInstance[]>;
    list(params: BrandVettingListInstanceOptions, callback?: (error: Error | null, items: BrandVettingInstance[]) => any): Promise<BrandVettingInstance[]>;
    /**
     * Retrieve a single page of BrandVettingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BrandVettingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: BrandVettingPage) => any): Promise<BrandVettingPage>;
    page(params: BrandVettingListInstancePageOptions, callback?: (error: Error | null, items: BrandVettingPage) => any): Promise<BrandVettingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BrandVettingListInstance(version: V1, brandSid: string): BrandVettingListInstance;
export declare class BrandVettingPage extends Page<V1, BrandVettingPayload, BrandVettingResource, BrandVettingInstance> {
    /**
     * Initialize the BrandVettingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: BrandVettingSolution);
    /**
     * Build an instance of BrandVettingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: BrandVettingResource): BrandVettingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
