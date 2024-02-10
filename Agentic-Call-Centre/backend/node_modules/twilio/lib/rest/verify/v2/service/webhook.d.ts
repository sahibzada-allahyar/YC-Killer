/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
export type WebhookMethods = "GET" | "POST";
export type WebhookStatus = "enabled" | "disabled";
export type WebhookVersion = "v1" | "v2";
/**
 * Options to pass to update a WebhookInstance
 */
export interface WebhookContextUpdateOptions {
    /** The string that you assigned to describe the webhook. **This value should not contain PII.** */
    friendlyName?: string;
    /** The array of events that this Webhook is subscribed to. Possible event types: `*, factor.deleted, factor.created, factor.verified, challenge.approved, challenge.denied` */
    eventTypes?: Array<string>;
    /** The URL associated with this Webhook. */
    webhookUrl?: string;
    /**  */
    status?: WebhookStatus;
    /**  */
    version?: WebhookVersion;
}
/**
 * Options to pass to create a WebhookInstance
 */
export interface WebhookListInstanceCreateOptions {
    /** The string that you assigned to describe the webhook. **This value should not contain PII.** */
    friendlyName: string;
    /** The array of events that this Webhook is subscribed to. Possible event types: `*, factor.deleted, factor.created, factor.verified, challenge.approved, challenge.denied` */
    eventTypes: Array<string>;
    /** The URL associated with this Webhook. */
    webhookUrl: string;
    /**  */
    status?: WebhookStatus;
    /**  */
    version?: WebhookVersion;
}
/**
 * Options to pass to each
 */
export interface WebhookListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: WebhookInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface WebhookListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface WebhookListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface WebhookContext {
    /**
     * Remove a WebhookInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a WebhookInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebhookInstance
     */
    fetch(callback?: (error: Error | null, item?: WebhookInstance) => any): Promise<WebhookInstance>;
    /**
     * Update a WebhookInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebhookInstance
     */
    update(callback?: (error: Error | null, item?: WebhookInstance) => any): Promise<WebhookInstance>;
    /**
     * Update a WebhookInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebhookInstance
     */
    update(params: WebhookContextUpdateOptions, callback?: (error: Error | null, item?: WebhookInstance) => any): Promise<WebhookInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WebhookContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class WebhookContextImpl implements WebhookContext {
    protected _version: V2;
    protected _solution: WebhookContextSolution;
    protected _uri: string;
    constructor(_version: V2, serviceSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: WebhookInstance) => any): Promise<WebhookInstance>;
    update(params?: WebhookContextUpdateOptions | ((error: Error | null, item?: WebhookInstance) => any), callback?: (error: Error | null, item?: WebhookInstance) => any): Promise<WebhookInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WebhookContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WebhookPayload extends TwilioResponsePayload {
    webhooks: WebhookResource[];
}
interface WebhookResource {
    sid: string;
    service_sid: string;
    account_sid: string;
    friendly_name: string;
    event_types: Array<string>;
    status: WebhookStatus;
    version: WebhookVersion;
    webhook_url: string;
    webhook_method: WebhookMethods;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class WebhookInstance {
    protected _version: V2;
    protected _solution: WebhookContextSolution;
    protected _context?: WebhookContext;
    constructor(_version: V2, payload: WebhookResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Webhook resource.
     */
    sid: string;
    /**
     * The unique SID identifier of the Service.
     */
    serviceSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Service resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the webhook. **This value should not contain PII.**
     */
    friendlyName: string;
    /**
     * The array of events that this Webhook is subscribed to. Possible event types: `*, factor.deleted, factor.created, factor.verified, challenge.approved, challenge.denied`
     */
    eventTypes: Array<string>;
    status: WebhookStatus;
    version: WebhookVersion;
    /**
     * The URL associated with this Webhook.
     */
    webhookUrl: string;
    webhookMethod: WebhookMethods;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Webhook resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a WebhookInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a WebhookInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebhookInstance
     */
    fetch(callback?: (error: Error | null, item?: WebhookInstance) => any): Promise<WebhookInstance>;
    /**
     * Update a WebhookInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebhookInstance
     */
    update(callback?: (error: Error | null, item?: WebhookInstance) => any): Promise<WebhookInstance>;
    /**
     * Update a WebhookInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebhookInstance
     */
    update(params: WebhookContextUpdateOptions, callback?: (error: Error | null, item?: WebhookInstance) => any): Promise<WebhookInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        serviceSid: string;
        accountSid: string;
        friendlyName: string;
        eventTypes: string[];
        status: WebhookStatus;
        version: WebhookVersion;
        webhookUrl: string;
        webhookMethod: WebhookMethods;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WebhookSolution {
    serviceSid: string;
}
export interface WebhookListInstance {
    _version: V2;
    _solution: WebhookSolution;
    _uri: string;
    (sid: string): WebhookContext;
    get(sid: string): WebhookContext;
    /**
     * Create a WebhookInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebhookInstance
     */
    create(params: WebhookListInstanceCreateOptions, callback?: (error: Error | null, item?: WebhookInstance) => any): Promise<WebhookInstance>;
    /**
     * Streams WebhookInstance records from the API.
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
     * @param { WebhookListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: WebhookInstance, done: (err?: Error) => void) => void): void;
    each(params: WebhookListInstanceEachOptions, callback?: (item: WebhookInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of WebhookInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: WebhookPage) => any): Promise<WebhookPage>;
    /**
     * Lists WebhookInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { WebhookListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: WebhookInstance[]) => any): Promise<WebhookInstance[]>;
    list(params: WebhookListInstanceOptions, callback?: (error: Error | null, items: WebhookInstance[]) => any): Promise<WebhookInstance[]>;
    /**
     * Retrieve a single page of WebhookInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { WebhookListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: WebhookPage) => any): Promise<WebhookPage>;
    page(params: WebhookListInstancePageOptions, callback?: (error: Error | null, items: WebhookPage) => any): Promise<WebhookPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WebhookListInstance(version: V2, serviceSid: string): WebhookListInstance;
export declare class WebhookPage extends Page<V2, WebhookPayload, WebhookResource, WebhookInstance> {
    /**
     * Initialize the WebhookPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: WebhookSolution);
    /**
     * Build an instance of WebhookInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: WebhookResource): WebhookInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
