/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
export type WebhookMethod = "GET" | "POST";
export type WebhookTarget = "webhook" | "trigger" | "studio";
/**
 * Options to pass to update a WebhookInstance
 */
export interface WebhookContextUpdateOptions {
    /** The absolute url the webhook request should be sent to. */
    "configuration.url"?: string;
    /**  */
    "configuration.method"?: WebhookMethod;
    /** The list of events, firing webhook event for this Conversation. */
    "configuration.filters"?: Array<string>;
    /** The list of keywords, firing webhook event for this Conversation. */
    "configuration.triggers"?: Array<string>;
    /** The studio flow SID, where the webhook should be sent to. */
    "configuration.flowSid"?: string;
}
/**
 * Options to pass to create a WebhookInstance
 */
export interface WebhookListInstanceCreateOptions {
    /**  */
    target: WebhookTarget;
    /** The absolute url the webhook request should be sent to. */
    "configuration.url"?: string;
    /**  */
    "configuration.method"?: WebhookMethod;
    /** The list of events, firing webhook event for this Conversation. */
    "configuration.filters"?: Array<string>;
    /** The list of keywords, firing webhook event for this Conversation. */
    "configuration.triggers"?: Array<string>;
    /** The studio flow SID, where the webhook should be sent to. */
    "configuration.flowSid"?: string;
    /** The message index for which and it\\\'s successors the webhook will be replayed. Not set by default */
    "configuration.replayAfter"?: number;
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
    chatServiceSid: string;
    conversationSid: string;
    sid: string;
}
export declare class WebhookContextImpl implements WebhookContext {
    protected _version: V1;
    protected _solution: WebhookContextSolution;
    protected _uri: string;
    constructor(_version: V1, chatServiceSid: string, conversationSid: string, sid: string);
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
    account_sid: string;
    chat_service_sid: string;
    conversation_sid: string;
    target: string;
    url: string;
    configuration: any;
    date_created: Date;
    date_updated: Date;
}
export declare class WebhookInstance {
    protected _version: V1;
    protected _solution: WebhookContextSolution;
    protected _context?: WebhookContext;
    constructor(_version: V1, payload: WebhookResource, chatServiceSid: string, conversationSid: string, sid?: string);
    /**
     * A 34 character string that uniquely identifies this resource.
     */
    sid: string;
    /**
     * The unique ID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this conversation.
     */
    accountSid: string;
    /**
     * The SID of the [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource) the Participant resource is associated with.
     */
    chatServiceSid: string;
    /**
     * The unique ID of the [Conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) for this webhook.
     */
    conversationSid: string;
    /**
     * The target of this webhook: `webhook`, `studio`, `trigger`
     */
    target: string;
    /**
     * An absolute API resource URL for this webhook.
     */
    url: string;
    /**
     * The configuration of this webhook. Is defined based on target.
     */
    configuration: any;
    /**
     * The date that this resource was created.
     */
    dateCreated: Date;
    /**
     * The date that this resource was last updated.
     */
    dateUpdated: Date;
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
        accountSid: string;
        chatServiceSid: string;
        conversationSid: string;
        target: string;
        url: string;
        configuration: any;
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WebhookSolution {
    chatServiceSid: string;
    conversationSid: string;
}
export interface WebhookListInstance {
    _version: V1;
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
export declare function WebhookListInstance(version: V1, chatServiceSid: string, conversationSid: string): WebhookListInstance;
export declare class WebhookPage extends Page<V1, WebhookPayload, WebhookResource, WebhookInstance> {
    /**
     * Initialize the WebhookPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: WebhookSolution);
    /**
     * Build an instance of WebhookInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: WebhookResource): WebhookInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
