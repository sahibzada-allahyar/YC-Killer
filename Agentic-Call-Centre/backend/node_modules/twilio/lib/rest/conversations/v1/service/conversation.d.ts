/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { MessageListInstance } from "./conversation/message";
import { ParticipantListInstance } from "./conversation/participant";
import { WebhookListInstance } from "./conversation/webhook";
export type ConversationState = "inactive" | "active" | "closed";
export type ConversationWebhookEnabledType = "true" | "false";
/**
 * Options to pass to remove a ConversationInstance
 */
export interface ConversationContextRemoveOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: ConversationWebhookEnabledType;
}
/**
 * Options to pass to update a ConversationInstance
 */
export interface ConversationContextUpdateOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: ConversationWebhookEnabledType;
    /** The human-readable name of this conversation, limited to 256 characters. Optional. */
    friendlyName?: string;
    /** The date that this resource was created. */
    dateCreated?: Date;
    /** The date that this resource was last updated. */
    dateUpdated?: Date;
    /** An optional string metadata field you can use to store any data you wish. The string value must contain structurally valid JSON if specified.  **Note** that if the attributes are not set \\\"{}\\\" will be returned. */
    attributes?: string;
    /** The unique ID of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) this conversation belongs to. */
    messagingServiceSid?: string;
    /**  */
    state?: ConversationState;
    /** ISO8601 duration when conversation will be switched to `inactive` state. Minimum value for this timer is 1 minute. */
    "timers.inactive"?: string;
    /** ISO8601 duration when conversation will be switched to `closed` state. Minimum value for this timer is 10 minutes. */
    "timers.closed"?: string;
    /** An application-defined string that uniquely identifies the resource. It can be used to address the resource in place of the resource\\\'s `sid` in the URL. */
    uniqueName?: string;
    /** The default email address that will be used when sending outbound emails in this conversation. */
    "bindings.email.address"?: string;
    /** The default name that will be used when sending outbound emails in this conversation. */
    "bindings.email.name"?: string;
}
/**
 * Options to pass to create a ConversationInstance
 */
export interface ConversationListInstanceCreateOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: ConversationWebhookEnabledType;
    /** The human-readable name of this conversation, limited to 256 characters. Optional. */
    friendlyName?: string;
    /** An application-defined string that uniquely identifies the resource. It can be used to address the resource in place of the resource\\\'s `sid` in the URL. */
    uniqueName?: string;
    /** An optional string metadata field you can use to store any data you wish. The string value must contain structurally valid JSON if specified.  **Note** that if the attributes are not set \\\"{}\\\" will be returned. */
    attributes?: string;
    /** The unique ID of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) this conversation belongs to. */
    messagingServiceSid?: string;
    /** The date that this resource was created. */
    dateCreated?: Date;
    /** The date that this resource was last updated. */
    dateUpdated?: Date;
    /**  */
    state?: ConversationState;
    /** ISO8601 duration when conversation will be switched to `inactive` state. Minimum value for this timer is 1 minute. */
    "timers.inactive"?: string;
    /** ISO8601 duration when conversation will be switched to `closed` state. Minimum value for this timer is 10 minutes. */
    "timers.closed"?: string;
    /** The default email address that will be used when sending outbound emails in this conversation. */
    "bindings.email.address"?: string;
    /** The default name that will be used when sending outbound emails in this conversation. */
    "bindings.email.name"?: string;
}
/**
 * Options to pass to each
 */
export interface ConversationListInstanceEachOptions {
    /** Start date or time in ISO8601 format for filtering list of Conversations. If a date is provided, the start time of the date is used (YYYY-MM-DDT00:00:00Z). Can be combined with other filters. */
    startDate?: string;
    /** End date or time in ISO8601 format for filtering list of Conversations. If a date is provided, the end time of the date is used (YYYY-MM-DDT23:59:59Z). Can be combined with other filters. */
    endDate?: string;
    /** State for sorting and filtering list of Conversations. Can be `active`, `inactive` or `closed` */
    state?: ConversationState;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ConversationInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ConversationListInstanceOptions {
    /** Start date or time in ISO8601 format for filtering list of Conversations. If a date is provided, the start time of the date is used (YYYY-MM-DDT00:00:00Z). Can be combined with other filters. */
    startDate?: string;
    /** End date or time in ISO8601 format for filtering list of Conversations. If a date is provided, the end time of the date is used (YYYY-MM-DDT23:59:59Z). Can be combined with other filters. */
    endDate?: string;
    /** State for sorting and filtering list of Conversations. Can be `active`, `inactive` or `closed` */
    state?: ConversationState;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ConversationListInstancePageOptions {
    /** Start date or time in ISO8601 format for filtering list of Conversations. If a date is provided, the start time of the date is used (YYYY-MM-DDT00:00:00Z). Can be combined with other filters. */
    startDate?: string;
    /** End date or time in ISO8601 format for filtering list of Conversations. If a date is provided, the end time of the date is used (YYYY-MM-DDT23:59:59Z). Can be combined with other filters. */
    endDate?: string;
    /** State for sorting and filtering list of Conversations. Can be `active`, `inactive` or `closed` */
    state?: ConversationState;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ConversationContext {
    messages: MessageListInstance;
    participants: ParticipantListInstance;
    webhooks: WebhookListInstance;
    /**
     * Remove a ConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a ConversationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConversationInstance
     */
    remove(params: ConversationContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConversationInstance
     */
    fetch(callback?: (error: Error | null, item?: ConversationInstance) => any): Promise<ConversationInstance>;
    /**
     * Update a ConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConversationInstance
     */
    update(callback?: (error: Error | null, item?: ConversationInstance) => any): Promise<ConversationInstance>;
    /**
     * Update a ConversationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConversationInstance
     */
    update(params: ConversationContextUpdateOptions, callback?: (error: Error | null, item?: ConversationInstance) => any): Promise<ConversationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ConversationContextSolution {
    chatServiceSid: string;
    sid: string;
}
export declare class ConversationContextImpl implements ConversationContext {
    protected _version: V1;
    protected _solution: ConversationContextSolution;
    protected _uri: string;
    protected _messages?: MessageListInstance;
    protected _participants?: ParticipantListInstance;
    protected _webhooks?: WebhookListInstance;
    constructor(_version: V1, chatServiceSid: string, sid: string);
    get messages(): MessageListInstance;
    get participants(): ParticipantListInstance;
    get webhooks(): WebhookListInstance;
    remove(params?: ConversationContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ConversationInstance) => any): Promise<ConversationInstance>;
    update(params?: ConversationContextUpdateOptions | ((error: Error | null, item?: ConversationInstance) => any), callback?: (error: Error | null, item?: ConversationInstance) => any): Promise<ConversationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ConversationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ConversationPayload extends TwilioResponsePayload {
    conversations: ConversationResource[];
}
interface ConversationResource {
    account_sid: string;
    chat_service_sid: string;
    messaging_service_sid: string;
    sid: string;
    friendly_name: string;
    unique_name: string;
    attributes: string;
    state: ConversationState;
    date_created: Date;
    date_updated: Date;
    timers: any;
    url: string;
    links: Record<string, string>;
    bindings: any;
}
export declare class ConversationInstance {
    protected _version: V1;
    protected _solution: ConversationContextSolution;
    protected _context?: ConversationContext;
    constructor(_version: V1, payload: ConversationResource, chatServiceSid: string, sid?: string);
    /**
     * The unique ID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this conversation.
     */
    accountSid: string;
    /**
     * The unique ID of the [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource) this conversation belongs to.
     */
    chatServiceSid: string;
    /**
     * The unique ID of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) this conversation belongs to.
     */
    messagingServiceSid: string;
    /**
     * A 34 character string that uniquely identifies this resource.
     */
    sid: string;
    /**
     * The human-readable name of this conversation, limited to 256 characters. Optional.
     */
    friendlyName: string;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used to address the resource in place of the resource\'s `sid` in the URL.
     */
    uniqueName: string;
    /**
     * An optional string metadata field you can use to store any data you wish. The string value must contain structurally valid JSON if specified.  **Note** that if the attributes are not set \"{}\" will be returned.
     */
    attributes: string;
    state: ConversationState;
    /**
     * The date that this resource was created.
     */
    dateCreated: Date;
    /**
     * The date that this resource was last updated.
     */
    dateUpdated: Date;
    /**
     * Timer date values representing state update for this conversation.
     */
    timers: any;
    /**
     * An absolute API resource URL for this conversation.
     */
    url: string;
    /**
     * Contains absolute URLs to access the [participants](https://www.twilio.com/docs/conversations/api/conversation-participant-resource), [messages](https://www.twilio.com/docs/conversations/api/conversation-message-resource) and [webhooks](https://www.twilio.com/docs/conversations/api/conversation-scoped-webhook-resource) of this conversation.
     */
    links: Record<string, string>;
    bindings: any;
    private get _proxy();
    /**
     * Remove a ConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a ConversationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConversationInstance
     */
    remove(params: ConversationContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConversationInstance
     */
    fetch(callback?: (error: Error | null, item?: ConversationInstance) => any): Promise<ConversationInstance>;
    /**
     * Update a ConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConversationInstance
     */
    update(callback?: (error: Error | null, item?: ConversationInstance) => any): Promise<ConversationInstance>;
    /**
     * Update a ConversationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConversationInstance
     */
    update(params: ConversationContextUpdateOptions, callback?: (error: Error | null, item?: ConversationInstance) => any): Promise<ConversationInstance>;
    /**
     * Access the messages.
     */
    messages(): MessageListInstance;
    /**
     * Access the participants.
     */
    participants(): ParticipantListInstance;
    /**
     * Access the webhooks.
     */
    webhooks(): WebhookListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        chatServiceSid: string;
        messagingServiceSid: string;
        sid: string;
        friendlyName: string;
        uniqueName: string;
        attributes: string;
        state: ConversationState;
        dateCreated: Date;
        dateUpdated: Date;
        timers: any;
        url: string;
        links: Record<string, string>;
        bindings: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ConversationSolution {
    chatServiceSid: string;
}
export interface ConversationListInstance {
    _version: V1;
    _solution: ConversationSolution;
    _uri: string;
    (sid: string): ConversationContext;
    get(sid: string): ConversationContext;
    /**
     * Create a ConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConversationInstance
     */
    create(callback?: (error: Error | null, item?: ConversationInstance) => any): Promise<ConversationInstance>;
    /**
     * Create a ConversationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConversationInstance
     */
    create(params: ConversationListInstanceCreateOptions, callback?: (error: Error | null, item?: ConversationInstance) => any): Promise<ConversationInstance>;
    /**
     * Streams ConversationInstance records from the API.
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
     * @param { ConversationListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ConversationInstance, done: (err?: Error) => void) => void): void;
    each(params: ConversationListInstanceEachOptions, callback?: (item: ConversationInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ConversationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ConversationPage) => any): Promise<ConversationPage>;
    /**
     * Lists ConversationInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ConversationListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ConversationInstance[]) => any): Promise<ConversationInstance[]>;
    list(params: ConversationListInstanceOptions, callback?: (error: Error | null, items: ConversationInstance[]) => any): Promise<ConversationInstance[]>;
    /**
     * Retrieve a single page of ConversationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ConversationListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ConversationPage) => any): Promise<ConversationPage>;
    page(params: ConversationListInstancePageOptions, callback?: (error: Error | null, items: ConversationPage) => any): Promise<ConversationPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ConversationListInstance(version: V1, chatServiceSid: string): ConversationListInstance;
export declare class ConversationPage extends Page<V1, ConversationPayload, ConversationResource, ConversationInstance> {
    /**
     * Initialize the ConversationPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ConversationSolution);
    /**
     * Build an instance of ConversationInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ConversationResource): ConversationInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
