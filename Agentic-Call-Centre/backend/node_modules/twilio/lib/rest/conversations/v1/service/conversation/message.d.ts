/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
import { DeliveryReceiptListInstance } from "./message/deliveryReceipt";
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
    /** The channel specific identifier of the message\\\'s author. Defaults to `system`. */
    author?: string;
    /** The content of the message, can be up to 1,600 characters long. */
    body?: string;
    /** The date that this resource was created. */
    dateCreated?: Date;
    /** The date that this resource was last updated. `null` if the message has not been edited. */
    dateUpdated?: Date;
    /** A string metadata field you can use to store any data you wish. The string value must contain structurally valid JSON if specified.  **Note** that if the attributes are not set \\\"{}\\\" will be returned. */
    attributes?: string;
    /** The subject of the message, can be up to 256 characters long. */
    subject?: string;
}
/**
 * Options to pass to create a MessageInstance
 */
export interface MessageListInstanceCreateOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: MessageWebhookEnabledType;
    /** The channel specific identifier of the message\\\'s author. Defaults to `system`. */
    author?: string;
    /** The content of the message, can be up to 1,600 characters long. */
    body?: string;
    /** The date that this resource was created. */
    dateCreated?: Date;
    /** The date that this resource was last updated. `null` if the message has not been edited. */
    dateUpdated?: Date;
    /** A string metadata field you can use to store any data you wish. The string value must contain structurally valid JSON if specified.  **Note** that if the attributes are not set \\\"{}\\\" will be returned. */
    attributes?: string;
    /** The Media SID to be attached to the new Message. */
    mediaSid?: string;
    /** The unique ID of the multi-channel [Rich Content](https://www.twilio.com/docs/content) template, required for template-generated messages.  **Note** that if this field is set, `Body` and `MediaSid` parameters are ignored. */
    contentSid?: string;
    /** A structurally valid JSON string that contains values to resolve Rich Content template variables. */
    contentVariables?: string;
    /** The subject of the message, can be up to 256 characters long. */
    subject?: string;
}
/**
 * Options to pass to each
 */
export interface MessageListInstanceEachOptions {
    /** The sort order of the returned messages. Can be: `asc` (ascending) or `desc` (descending), with `asc` as the default. */
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
    /** The sort order of the returned messages. Can be: `asc` (ascending) or `desc` (descending), with `asc` as the default. */
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
    /** The sort order of the returned messages. Can be: `asc` (ascending) or `desc` (descending), with `asc` as the default. */
    order?: MessageOrderType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface MessageContext {
    deliveryReceipts: DeliveryReceiptListInstance;
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
    chatServiceSid: string;
    conversationSid: string;
    sid: string;
}
export declare class MessageContextImpl implements MessageContext {
    protected _version: V1;
    protected _solution: MessageContextSolution;
    protected _uri: string;
    protected _deliveryReceipts?: DeliveryReceiptListInstance;
    constructor(_version: V1, chatServiceSid: string, conversationSid: string, sid: string);
    get deliveryReceipts(): DeliveryReceiptListInstance;
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
    account_sid: string;
    chat_service_sid: string;
    conversation_sid: string;
    sid: string;
    index: number;
    author: string;
    body: string;
    media: Array<any>;
    attributes: string;
    participant_sid: string;
    date_created: Date;
    date_updated: Date;
    delivery: any;
    url: string;
    links: Record<string, string>;
    content_sid: string;
}
export declare class MessageInstance {
    protected _version: V1;
    protected _solution: MessageContextSolution;
    protected _context?: MessageContext;
    constructor(_version: V1, payload: MessageResource, chatServiceSid: string, conversationSid: string, sid?: string);
    /**
     * The unique ID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this message.
     */
    accountSid: string;
    /**
     * The SID of the [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource) the Participant resource is associated with.
     */
    chatServiceSid: string;
    /**
     * The unique ID of the [Conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) for this message.
     */
    conversationSid: string;
    /**
     * A 34 character string that uniquely identifies this resource.
     */
    sid: string;
    /**
     * The index of the message within the [Conversation](https://www.twilio.com/docs/conversations/api/conversation-resource).
     */
    index: number;
    /**
     * The channel specific identifier of the message\'s author. Defaults to `system`.
     */
    author: string;
    /**
     * The content of the message, can be up to 1,600 characters long.
     */
    body: string;
    /**
     * An array of objects that describe the Message\'s media, if the message contains media. Each object contains these fields: `content_type` with the MIME type of the media, `filename` with the name of the media, `sid` with the SID of the Media resource, and `size` with the media object\'s file size in bytes. If the Message has no media, this value is `null`.
     */
    media: Array<any>;
    /**
     * A string metadata field you can use to store any data you wish. The string value must contain structurally valid JSON if specified.  **Note** that if the attributes are not set \"{}\" will be returned.
     */
    attributes: string;
    /**
     * The unique ID of messages\'s author participant. Null in case of `system` sent message.
     */
    participantSid: string;
    /**
     * The date that this resource was created.
     */
    dateCreated: Date;
    /**
     * The date that this resource was last updated. `null` if the message has not been edited.
     */
    dateUpdated: Date;
    /**
     * An object that contains the summary of delivery statuses for the message to non-chat participants.
     */
    delivery: any;
    /**
     * An absolute API resource URL for this message.
     */
    url: string;
    /**
     * Contains an absolute API resource URL to access the delivery & read receipts of this message.
     */
    links: Record<string, string>;
    /**
     * The unique ID of the multi-channel [Rich Content](https://www.twilio.com/docs/content) template.
     */
    contentSid: string;
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
     * Access the deliveryReceipts.
     */
    deliveryReceipts(): DeliveryReceiptListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        chatServiceSid: string;
        conversationSid: string;
        sid: string;
        index: number;
        author: string;
        body: string;
        media: any[];
        attributes: string;
        participantSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        delivery: any;
        url: string;
        links: Record<string, string>;
        contentSid: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface MessageSolution {
    chatServiceSid: string;
    conversationSid: string;
}
export interface MessageListInstance {
    _version: V1;
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
export declare function MessageListInstance(version: V1, chatServiceSid: string, conversationSid: string): MessageListInstance;
export declare class MessagePage extends Page<V1, MessagePayload, MessageResource, MessageInstance> {
    /**
     * Initialize the MessagePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: MessageSolution);
    /**
     * Build an instance of MessageInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: MessageResource): MessageInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
