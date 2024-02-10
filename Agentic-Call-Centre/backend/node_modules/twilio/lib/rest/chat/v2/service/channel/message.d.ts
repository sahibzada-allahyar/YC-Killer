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
    /** The message to send to the channel. Can be an empty string or `null`, which sets the value as an empty string. You can send structured data in the body by serializing it as a string. */
    body?: string;
    /** A valid JSON string that contains application-specific data. */
    attributes?: string;
    /** The date, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format, to assign to the resource as the date it was created. The default value is the current time set by the Chat service. This parameter should only be used when a Chat\\\'s history is being recreated from a backup/separate source. */
    dateCreated?: Date;
    /** The date, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format, to assign to the resource as the date it was last updated. */
    dateUpdated?: Date;
    /** The [Identity](https://www.twilio.com/docs/chat/identity) of the User who last updated the Message, if applicable. */
    lastUpdatedBy?: string;
    /** The [Identity](https://www.twilio.com/docs/chat/identity) of the message\\\'s author. */
    from?: string;
}
/**
 * Options to pass to create a MessageInstance
 */
export interface MessageListInstanceCreateOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: MessageWebhookEnabledType;
    /** The [Identity](https://www.twilio.com/docs/chat/identity) of the new message\\\'s author. The default value is `system`. */
    from?: string;
    /** A valid JSON string that contains application-specific data. */
    attributes?: string;
    /** The date, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format, to assign to the resource as the date it was created. The default value is the current time set by the Chat service. This parameter should only be used when a Chat\\\'s history is being recreated from a backup/separate source. */
    dateCreated?: Date;
    /** The date, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format, to assign to the resource as the date it was last updated. */
    dateUpdated?: Date;
    /** The [Identity](https://www.twilio.com/docs/chat/identity) of the User who last updated the Message, if applicable. */
    lastUpdatedBy?: string;
    /** The message to send to the channel. Can be an empty string or `null`, which sets the value as an empty string. You can send structured data in the body by serializing it as a string. */
    body?: string;
    /** The SID of the [Media](https://www.twilio.com/docs/chat/rest/media) to attach to the new Message. */
    mediaSid?: string;
}
/**
 * Options to pass to each
 */
export interface MessageListInstanceEachOptions {
    /** The sort order of the returned messages. Can be: `asc` (ascending) or `desc` (descending) with `asc` as the default. */
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
    /** The sort order of the returned messages. Can be: `asc` (ascending) or `desc` (descending) with `asc` as the default. */
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
    /** The sort order of the returned messages. Can be: `asc` (ascending) or `desc` (descending) with `asc` as the default. */
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
    /**
     * The unique string that we created to identify the Message resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Message resource.
     */
    accountSid: string;
    /**
     * The JSON string that stores application-specific data. If attributes have not been set, `{}` is returned.
     */
    attributes: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/chat/rest/service-resource) the Message resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the [Channel](https://www.twilio.com/docs/chat/channels) that the message was sent to.
     */
    to: string;
    /**
     * The SID of the [Channel](https://www.twilio.com/docs/chat/channels) the Message resource belongs to.
     */
    channelSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The [Identity](https://www.twilio.com/docs/chat/identity) of the User who last updated the Message, if applicable.
     */
    lastUpdatedBy: string;
    /**
     * Whether the message has been edited since it was created.
     */
    wasEdited: boolean;
    /**
     * The [Identity](https://www.twilio.com/docs/chat/identity) of the message\'s author. The default value is `system`.
     */
    from: string;
    /**
     * The content of the message.
     */
    body: string;
    /**
     * The index of the message within the [Channel](https://www.twilio.com/docs/chat/channels). Indices may skip numbers, but will always be in order of when the message was received.
     */
    index: number;
    /**
     * The Message type. Can be: `text` or `media`.
     */
    type: string;
    /**
     * An object that describes the Message\'s media, if the message contains media. The object contains these fields: `content_type` with the MIME type of the media, `filename` with the name of the media, `sid` with the SID of the Media resource, and `size` with the media object\'s file size in bytes. If the Message has no media, this value is `null`.
     */
    media: any;
    /**
     * The absolute URL of the Message resource.
     */
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
