/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type WebChannelChatStatus = "inactive";
/**
 * Options to pass to update a WebChannelInstance
 */
export interface WebChannelContextUpdateOptions {
    /**  */
    chatStatus?: WebChannelChatStatus;
    /** The post-engagement data. */
    postEngagementData?: string;
}
/**
 * Options to pass to create a WebChannelInstance
 */
export interface WebChannelListInstanceCreateOptions {
    /** The SID of the Flex Flow. */
    flexFlowSid: string;
    /** The chat identity. */
    identity: string;
    /** The chat participant\\\'s friendly name. */
    customerFriendlyName: string;
    /** The chat channel\\\'s friendly name. */
    chatFriendlyName: string;
    /** The chat channel\\\'s unique name. */
    chatUniqueName?: string;
    /** The pre-engagement data. */
    preEngagementData?: string;
}
/**
 * Options to pass to each
 */
export interface WebChannelListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: WebChannelInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface WebChannelListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface WebChannelListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface WebChannelContext {
    /**
     * Remove a WebChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a WebChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: WebChannelInstance) => any): Promise<WebChannelInstance>;
    /**
     * Update a WebChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebChannelInstance
     */
    update(callback?: (error: Error | null, item?: WebChannelInstance) => any): Promise<WebChannelInstance>;
    /**
     * Update a WebChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebChannelInstance
     */
    update(params: WebChannelContextUpdateOptions, callback?: (error: Error | null, item?: WebChannelInstance) => any): Promise<WebChannelInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WebChannelContextSolution {
    sid: string;
}
export declare class WebChannelContextImpl implements WebChannelContext {
    protected _version: V1;
    protected _solution: WebChannelContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: WebChannelInstance) => any): Promise<WebChannelInstance>;
    update(params?: WebChannelContextUpdateOptions | ((error: Error | null, item?: WebChannelInstance) => any), callback?: (error: Error | null, item?: WebChannelInstance) => any): Promise<WebChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WebChannelContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WebChannelPayload extends TwilioResponsePayload {
    flex_chat_channels: WebChannelResource[];
}
interface WebChannelResource {
    account_sid: string;
    flex_flow_sid: string;
    sid: string;
    url: string;
    date_created: Date;
    date_updated: Date;
}
export declare class WebChannelInstance {
    protected _version: V1;
    protected _solution: WebChannelContextSolution;
    protected _context?: WebChannelContext;
    constructor(_version: V1, payload: WebChannelResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the WebChannel resource and owns this Workflow.
     */
    accountSid: string;
    /**
     * The SID of the Flex Flow.
     */
    flexFlowSid: string;
    /**
     * The unique string that we created to identify the WebChannel resource.
     */
    sid: string;
    /**
     * The absolute URL of the WebChannel resource.
     */
    url: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    private get _proxy();
    /**
     * Remove a WebChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a WebChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: WebChannelInstance) => any): Promise<WebChannelInstance>;
    /**
     * Update a WebChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebChannelInstance
     */
    update(callback?: (error: Error | null, item?: WebChannelInstance) => any): Promise<WebChannelInstance>;
    /**
     * Update a WebChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebChannelInstance
     */
    update(params: WebChannelContextUpdateOptions, callback?: (error: Error | null, item?: WebChannelInstance) => any): Promise<WebChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        flexFlowSid: string;
        sid: string;
        url: string;
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WebChannelSolution {
}
export interface WebChannelListInstance {
    _version: V1;
    _solution: WebChannelSolution;
    _uri: string;
    (sid: string): WebChannelContext;
    get(sid: string): WebChannelContext;
    /**
     * Create a WebChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebChannelInstance
     */
    create(params: WebChannelListInstanceCreateOptions, callback?: (error: Error | null, item?: WebChannelInstance) => any): Promise<WebChannelInstance>;
    /**
     * Streams WebChannelInstance records from the API.
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
     * @param { WebChannelListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: WebChannelInstance, done: (err?: Error) => void) => void): void;
    each(params: WebChannelListInstanceEachOptions, callback?: (item: WebChannelInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of WebChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: WebChannelPage) => any): Promise<WebChannelPage>;
    /**
     * Lists WebChannelInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { WebChannelListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: WebChannelInstance[]) => any): Promise<WebChannelInstance[]>;
    list(params: WebChannelListInstanceOptions, callback?: (error: Error | null, items: WebChannelInstance[]) => any): Promise<WebChannelInstance[]>;
    /**
     * Retrieve a single page of WebChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { WebChannelListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: WebChannelPage) => any): Promise<WebChannelPage>;
    page(params: WebChannelListInstancePageOptions, callback?: (error: Error | null, items: WebChannelPage) => any): Promise<WebChannelPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WebChannelListInstance(version: V1): WebChannelListInstance;
export declare class WebChannelPage extends Page<V1, WebChannelPayload, WebChannelResource, WebChannelInstance> {
    /**
     * Initialize the WebChannelPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: WebChannelSolution);
    /**
     * Build an instance of WebChannelInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: WebChannelResource): WebChannelInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
