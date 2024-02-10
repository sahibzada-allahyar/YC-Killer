/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to each
 */
export interface ChannelSenderListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ChannelSenderInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ChannelSenderListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ChannelSenderListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ChannelSenderContext {
    /**
     * Fetch a ChannelSenderInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelSenderInstance
     */
    fetch(callback?: (error: Error | null, item?: ChannelSenderInstance) => any): Promise<ChannelSenderInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ChannelSenderContextSolution {
    messagingServiceSid: string;
    sid: string;
}
export declare class ChannelSenderContextImpl implements ChannelSenderContext {
    protected _version: V1;
    protected _solution: ChannelSenderContextSolution;
    protected _uri: string;
    constructor(_version: V1, messagingServiceSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: ChannelSenderInstance) => any): Promise<ChannelSenderInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ChannelSenderContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ChannelSenderPayload extends TwilioResponsePayload {
    senders: ChannelSenderResource[];
}
interface ChannelSenderResource {
    account_sid: string;
    messaging_service_sid: string;
    sid: string;
    sender: string;
    sender_type: string;
    country_code: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class ChannelSenderInstance {
    protected _version: V1;
    protected _solution: ChannelSenderContextSolution;
    protected _context?: ChannelSenderContext;
    constructor(_version: V1, payload: ChannelSenderResource, messagingServiceSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the ChannelSender resource.
     */
    accountSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/messaging/services) the resource is associated with.
     */
    messagingServiceSid: string;
    /**
     * The unique string that we created to identify the ChannelSender resource.
     */
    sid: string;
    /**
     * The unique string that identifies the sender e.g whatsapp:+123456XXXX.
     */
    sender: string;
    /**
     * A string value that identifies the sender type e.g WhatsApp, Messenger.
     */
    senderType: string;
    /**
     * The 2-character [ISO Country Code](https://www.iso.org/iso-3166-country-codes.html) of the number.
     */
    countryCode: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the ChannelSender resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a ChannelSenderInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelSenderInstance
     */
    fetch(callback?: (error: Error | null, item?: ChannelSenderInstance) => any): Promise<ChannelSenderInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        messagingServiceSid: string;
        sid: string;
        sender: string;
        senderType: string;
        countryCode: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ChannelSenderSolution {
    messagingServiceSid: string;
}
export interface ChannelSenderListInstance {
    _version: V1;
    _solution: ChannelSenderSolution;
    _uri: string;
    (sid: string): ChannelSenderContext;
    get(sid: string): ChannelSenderContext;
    /**
     * Streams ChannelSenderInstance records from the API.
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
     * @param { ChannelSenderListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ChannelSenderInstance, done: (err?: Error) => void) => void): void;
    each(params: ChannelSenderListInstanceEachOptions, callback?: (item: ChannelSenderInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ChannelSenderInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ChannelSenderPage) => any): Promise<ChannelSenderPage>;
    /**
     * Lists ChannelSenderInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ChannelSenderListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ChannelSenderInstance[]) => any): Promise<ChannelSenderInstance[]>;
    list(params: ChannelSenderListInstanceOptions, callback?: (error: Error | null, items: ChannelSenderInstance[]) => any): Promise<ChannelSenderInstance[]>;
    /**
     * Retrieve a single page of ChannelSenderInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ChannelSenderListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ChannelSenderPage) => any): Promise<ChannelSenderPage>;
    page(params: ChannelSenderListInstancePageOptions, callback?: (error: Error | null, items: ChannelSenderPage) => any): Promise<ChannelSenderPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ChannelSenderListInstance(version: V1, messagingServiceSid: string): ChannelSenderListInstance;
export declare class ChannelSenderPage extends Page<V1, ChannelSenderPayload, ChannelSenderResource, ChannelSenderInstance> {
    /**
     * Initialize the ChannelSenderPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ChannelSenderSolution);
    /**
     * Build an instance of ChannelSenderInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ChannelSenderResource): ChannelSenderInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
