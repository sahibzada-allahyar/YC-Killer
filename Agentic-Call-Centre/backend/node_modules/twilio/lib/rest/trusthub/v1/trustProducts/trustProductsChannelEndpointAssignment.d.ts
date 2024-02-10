/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to create a TrustProductsChannelEndpointAssignmentInstance
 */
export interface TrustProductsChannelEndpointAssignmentListInstanceCreateOptions {
    /** The type of channel endpoint. eg: phone-number */
    channelEndpointType: string;
    /** The SID of an channel endpoint */
    channelEndpointSid: string;
}
/**
 * Options to pass to each
 */
export interface TrustProductsChannelEndpointAssignmentListInstanceEachOptions {
    /** The SID of an channel endpoint */
    channelEndpointSid?: string;
    /** comma separated list of channel endpoint sids */
    channelEndpointSids?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TrustProductsChannelEndpointAssignmentInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TrustProductsChannelEndpointAssignmentListInstanceOptions {
    /** The SID of an channel endpoint */
    channelEndpointSid?: string;
    /** comma separated list of channel endpoint sids */
    channelEndpointSids?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TrustProductsChannelEndpointAssignmentListInstancePageOptions {
    /** The SID of an channel endpoint */
    channelEndpointSid?: string;
    /** comma separated list of channel endpoint sids */
    channelEndpointSids?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TrustProductsChannelEndpointAssignmentContext {
    /**
     * Remove a TrustProductsChannelEndpointAssignmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TrustProductsChannelEndpointAssignmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrustProductsChannelEndpointAssignmentInstance
     */
    fetch(callback?: (error: Error | null, item?: TrustProductsChannelEndpointAssignmentInstance) => any): Promise<TrustProductsChannelEndpointAssignmentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TrustProductsChannelEndpointAssignmentContextSolution {
    trustProductSid: string;
    sid: string;
}
export declare class TrustProductsChannelEndpointAssignmentContextImpl implements TrustProductsChannelEndpointAssignmentContext {
    protected _version: V1;
    protected _solution: TrustProductsChannelEndpointAssignmentContextSolution;
    protected _uri: string;
    constructor(_version: V1, trustProductSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: TrustProductsChannelEndpointAssignmentInstance) => any): Promise<TrustProductsChannelEndpointAssignmentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TrustProductsChannelEndpointAssignmentContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TrustProductsChannelEndpointAssignmentPayload extends TwilioResponsePayload {
    results: TrustProductsChannelEndpointAssignmentResource[];
}
interface TrustProductsChannelEndpointAssignmentResource {
    sid: string;
    trust_product_sid: string;
    account_sid: string;
    channel_endpoint_type: string;
    channel_endpoint_sid: string;
    date_created: Date;
    url: string;
}
export declare class TrustProductsChannelEndpointAssignmentInstance {
    protected _version: V1;
    protected _solution: TrustProductsChannelEndpointAssignmentContextSolution;
    protected _context?: TrustProductsChannelEndpointAssignmentContext;
    constructor(_version: V1, payload: TrustProductsChannelEndpointAssignmentResource, trustProductSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Item Assignment resource.
     */
    sid: string;
    /**
     * The unique string that we created to identify the CustomerProfile resource.
     */
    trustProductSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Item Assignment resource.
     */
    accountSid: string;
    /**
     * The type of channel endpoint. eg: phone-number
     */
    channelEndpointType: string;
    /**
     * The SID of an channel endpoint
     */
    channelEndpointSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The absolute URL of the Identity resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a TrustProductsChannelEndpointAssignmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TrustProductsChannelEndpointAssignmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrustProductsChannelEndpointAssignmentInstance
     */
    fetch(callback?: (error: Error | null, item?: TrustProductsChannelEndpointAssignmentInstance) => any): Promise<TrustProductsChannelEndpointAssignmentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        trustProductSid: string;
        accountSid: string;
        channelEndpointType: string;
        channelEndpointSid: string;
        dateCreated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TrustProductsChannelEndpointAssignmentSolution {
    trustProductSid: string;
}
export interface TrustProductsChannelEndpointAssignmentListInstance {
    _version: V1;
    _solution: TrustProductsChannelEndpointAssignmentSolution;
    _uri: string;
    (sid: string): TrustProductsChannelEndpointAssignmentContext;
    get(sid: string): TrustProductsChannelEndpointAssignmentContext;
    /**
     * Create a TrustProductsChannelEndpointAssignmentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrustProductsChannelEndpointAssignmentInstance
     */
    create(params: TrustProductsChannelEndpointAssignmentListInstanceCreateOptions, callback?: (error: Error | null, item?: TrustProductsChannelEndpointAssignmentInstance) => any): Promise<TrustProductsChannelEndpointAssignmentInstance>;
    /**
     * Streams TrustProductsChannelEndpointAssignmentInstance records from the API.
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
     * @param { TrustProductsChannelEndpointAssignmentListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TrustProductsChannelEndpointAssignmentInstance, done: (err?: Error) => void) => void): void;
    each(params: TrustProductsChannelEndpointAssignmentListInstanceEachOptions, callback?: (item: TrustProductsChannelEndpointAssignmentInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TrustProductsChannelEndpointAssignmentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TrustProductsChannelEndpointAssignmentPage) => any): Promise<TrustProductsChannelEndpointAssignmentPage>;
    /**
     * Lists TrustProductsChannelEndpointAssignmentInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TrustProductsChannelEndpointAssignmentListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TrustProductsChannelEndpointAssignmentInstance[]) => any): Promise<TrustProductsChannelEndpointAssignmentInstance[]>;
    list(params: TrustProductsChannelEndpointAssignmentListInstanceOptions, callback?: (error: Error | null, items: TrustProductsChannelEndpointAssignmentInstance[]) => any): Promise<TrustProductsChannelEndpointAssignmentInstance[]>;
    /**
     * Retrieve a single page of TrustProductsChannelEndpointAssignmentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TrustProductsChannelEndpointAssignmentListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TrustProductsChannelEndpointAssignmentPage) => any): Promise<TrustProductsChannelEndpointAssignmentPage>;
    page(params: TrustProductsChannelEndpointAssignmentListInstancePageOptions, callback?: (error: Error | null, items: TrustProductsChannelEndpointAssignmentPage) => any): Promise<TrustProductsChannelEndpointAssignmentPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TrustProductsChannelEndpointAssignmentListInstance(version: V1, trustProductSid: string): TrustProductsChannelEndpointAssignmentListInstance;
export declare class TrustProductsChannelEndpointAssignmentPage extends Page<V1, TrustProductsChannelEndpointAssignmentPayload, TrustProductsChannelEndpointAssignmentResource, TrustProductsChannelEndpointAssignmentInstance> {
    /**
     * Initialize the TrustProductsChannelEndpointAssignmentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: TrustProductsChannelEndpointAssignmentSolution);
    /**
     * Build an instance of TrustProductsChannelEndpointAssignmentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TrustProductsChannelEndpointAssignmentResource): TrustProductsChannelEndpointAssignmentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
