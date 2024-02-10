/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2 from "../../../V2";
export type MessageOrderType = "asc" | "desc";
export type MessageWebhookEnabledType = "true" | "false";
/**
 * Options to pass to remove a MessageInstance
 */
export interface MessageContextRemoveOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: MessageWebhookEnabledType;
}
/**
 * Options to pass to update a MessageInstance
 */
export interface MessageContextUpdateOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: MessageWebhookEnabledType;
    /**  */
    body?: string;
    /**  */
    attributes?: string;
    /**  */
    dateCreated?: Date;
    /**  */
    dateUpdated?: Date;
    /**  */
    lastUpdatedBy?: string;
    /**  */
    from?: string;
}
/**
 * Options to pass to create a MessageInstance
 */
export interface MessageListInstanceCreateOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: MessageWebhookEnabledType;
    /**  */
    from?: string;
    /**  */
    attributes?: string;
    /**  */
    dateCreated?: Date;
    /**  */
    dateUpdated?: Date;
    /**  */
    lastUpdatedBy?: string;
    /**  */
    body?: string;
    /**  */
    mediaSid?: string;
}
/**
 * Options to pass to each
 */
export interface MessageListInstanceEachOptions {
    /**  */
    order?: MessageOrderType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: MessageInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface MessageListInstanceOptions {
    /**  */
    order?: MessageOrderType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface MessageListInstancePageOptions {
    /**  */
    order?: MessageOrderType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface MessageContext {
    /**
     * Remove a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a MessageInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    remove(params: MessageContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    fetch(callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Update a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    update(callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Update a MessageInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    update(params: MessageContextUpdateOptions, callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface MessageContextSolution {
    serviceSid: string;
    channelSid: string;
    sid: string;
}
export declare class MessageContextImpl implements MessageContext {
    protected _version: V2;
    protected _solution: MessageContextSolution;
    protected _uri: string;
    constructor(_version: V2, serviceSid: string, channelSid: string, sid: string);
    remove(params?: MessageContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    update(params?: MessageContextUpdateOptions | ((error: Error | null, item?: MessageInstance) => any), callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): MessageContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface MessagePayload extends TwilioResponsePayload {
    messages: MessageResource[];
}
interface MessageResource {
    sid: string;
    account_sid: string;
    attributes: string;
    service_sid: string;
    to: string;
    channel_sid: string;
    date_created: Date;
    date_updated: Date;
    last_updated_by: string;
    was_edited: boolean;
    from: string;
    body: string;
    index: number;
    type: string;
    media: any;
    url: string;
}
export declare class MessageInstance {
    protected _version: V2;
    protected _solution: MessageContextSolution;
    protected _context?: MessageContext;
    constructor(_version: V2, payload: MessageResource, serviceSid: string, channelSid: string, sid?: string);
    sid: string;
    accountSid: string;
    attributes: string;
    serviceSid: string;
    to: string;
    channelSid: string;
    dateCreated: Date;
    dateUpdated: Date;
    lastUpdatedBy: string;
    wasEdited: boolean;
    from: string;
    body: string;
    index: number;
    type: string;
    media: any;
    url: string;
    private get _proxy();
    /**
     * Remove a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a MessageInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    remove(params: MessageContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    fetch(callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Update a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    update(callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Update a MessageInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    update(params: MessageContextUpdateOptions, callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        attributes: string;
        serviceSid: string;
        to: string;
        channelSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        lastUpdatedBy: string;
        wasEdited: boolean;
        from: string;
        body: string;
        index: number;
        type: string;
        media: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface MessageSolution {
    serviceSid: string;
    channelSid: string;
}
export interface MessageListInstance {
    _version: V2;
    _solution: MessageSolution;
    _uri: string;
    (sid: string): MessageContext;
    get(sid: string): MessageContext;
    /**
     * Create a MessageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    create(callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Create a MessageInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInstance
     */
    create(params: MessageListInstanceCreateOptions, callback?: (error: Error | null, item?: MessageInstance) => any): Promise<MessageInstance>;
    /**
     * Streams MessageInstance records from the API.
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
     * @param { MessageListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: MessageInstance, done: (err?: Error) => void) => void): void;
    each(params: MessageListInstanceEachOptions, callback?: (item: MessageInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of MessageInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: MessagePage) => any): Promise<MessagePage>;
    /**
     * Lists MessageInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MessageListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: MessageInstance[]) => any): Promise<MessageInstance[]>;
    list(params: MessageListInstanceOptions, callback?: (error: Error | null, items: MessageInstance[]) => any): Promise<MessageInstance[]>;
    /**
     * Retrieve a single page of MessageInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MessageListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: MessagePage) => any): Promise<MessagePage>;
    page(params: MessageListInstancePageOptions, callback?: (error: Error | null, items: MessagePage) => any): Promise<MessagePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function MessageListInstance(version: V2, serviceSid: string, channelSid: string): MessageListInstance;
export declare class MessagePage extends Page<V2, MessagePayload, MessageResource, MessageInstance> {
    /**
     * Initialize the MessagePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: MessageSolution);
    /**
     * Build an instance of MessageInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: MessageResource): MessageInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
