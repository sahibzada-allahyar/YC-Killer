/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
export type WebhookMethod = "GET" | "POST";
/**
 * Options to pass to update a WebhookInstance
 */
export interface WebhookContextUpdateOptions {
    /** The absolute url the pre-event webhook request should be sent to. */
    preWebhookUrl?: string;
    /** The absolute url the post-event webhook request should be sent to. */
    postWebhookUrl?: string;
    /** The list of events that your configured webhook targets will receive. Events not configured here will not fire. Possible values are `onParticipantAdd`, `onParticipantAdded`, `onDeliveryUpdated`, `onConversationUpdated`, `onConversationRemove`, `onParticipantRemove`, `onConversationUpdate`, `onMessageAdd`, `onMessageRemoved`, `onParticipantUpdated`, `onConversationAdded`, `onMessageAdded`, `onConversationAdd`, `onConversationRemoved`, `onParticipantUpdate`, `onMessageRemove`, `onMessageUpdated`, `onParticipantRemoved`, `onMessageUpdate` or `onConversationStateUpdated`. */
    filters?: Array<string>;
    /** The HTTP method to be used when sending a webhook request. One of `GET` or `POST`. */
    method?: string;
}
export interface WebhookContext {
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
}
export declare class WebhookContextImpl implements WebhookContext {
    protected _version: V1;
    protected _solution: WebhookContextSolution;
    protected _uri: string;
    constructor(_version: V1, chatServiceSid: string);
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
interface WebhookResource {
    account_sid: string;
    chat_service_sid: string;
    pre_webhook_url: string;
    post_webhook_url: string;
    filters: Array<string>;
    method: WebhookMethod;
    url: string;
}
export declare class WebhookInstance {
    protected _version: V1;
    protected _solution: WebhookContextSolution;
    protected _context?: WebhookContext;
    constructor(_version: V1, payload: WebhookResource, chatServiceSid: string);
    /**
     * The unique ID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this service.
     */
    accountSid: string;
    /**
     * The unique ID of the [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource) this conversation belongs to.
     */
    chatServiceSid: string;
    /**
     * The absolute url the pre-event webhook request should be sent to.
     */
    preWebhookUrl: string;
    /**
     * The absolute url the post-event webhook request should be sent to.
     */
    postWebhookUrl: string;
    /**
     * The list of events that your configured webhook targets will receive. Events not configured here will not fire. Possible values are `onParticipantAdd`, `onParticipantAdded`, `onDeliveryUpdated`, `onConversationUpdated`, `onConversationRemove`, `onParticipantRemove`, `onConversationUpdate`, `onMessageAdd`, `onMessageRemoved`, `onParticipantUpdated`, `onConversationAdded`, `onMessageAdded`, `onConversationAdd`, `onConversationRemoved`, `onParticipantUpdate`, `onMessageRemove`, `onMessageUpdated`, `onParticipantRemoved`, `onMessageUpdate` or `onConversationStateUpdated`.
     */
    filters: Array<string>;
    method: WebhookMethod;
    /**
     * An absolute API resource URL for this webhook.
     */
    url: string;
    private get _proxy();
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
        accountSid: string;
        chatServiceSid: string;
        preWebhookUrl: string;
        postWebhookUrl: string;
        filters: string[];
        method: WebhookMethod;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WebhookSolution {
    chatServiceSid: string;
}
export interface WebhookListInstance {
    _version: V1;
    _solution: WebhookSolution;
    _uri: string;
    (): WebhookContext;
    get(): WebhookContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WebhookListInstance(version: V1, chatServiceSid: string): WebhookListInstance;
export {};
