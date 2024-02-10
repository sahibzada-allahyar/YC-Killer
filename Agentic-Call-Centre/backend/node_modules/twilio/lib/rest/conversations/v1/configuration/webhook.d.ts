/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
export type WebhookMethod = "GET" | "POST";
export type WebhookTarget = "webhook" | "flex";
/**
 * Options to pass to update a WebhookInstance
 */
export interface WebhookContextUpdateOptions {
    /** The HTTP method to be used when sending a webhook request. */
    method?: string;
    /** The list of webhook event triggers that are enabled for this Service: `onMessageAdded`, `onMessageUpdated`, `onMessageRemoved`, `onConversationUpdated`, `onConversationRemoved`, `onParticipantAdded`, `onParticipantUpdated`, `onParticipantRemoved` */
    filters?: Array<string>;
    /** The absolute url the pre-event webhook request should be sent to. */
    preWebhookUrl?: string;
    /** The absolute url the post-event webhook request should be sent to. */
    postWebhookUrl?: string;
    /**  */
    target?: WebhookTarget;
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
}
export declare class WebhookContextImpl implements WebhookContext {
    protected _version: V1;
    protected _solution: WebhookContextSolution;
    protected _uri: string;
    constructor(_version: V1);
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
    method: WebhookMethod;
    filters: Array<string>;
    pre_webhook_url: string;
    post_webhook_url: string;
    target: WebhookTarget;
    url: string;
}
export declare class WebhookInstance {
    protected _version: V1;
    protected _solution: WebhookContextSolution;
    protected _context?: WebhookContext;
    constructor(_version: V1, payload: WebhookResource);
    /**
     * The unique ID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this conversation.
     */
    accountSid: string;
    method: WebhookMethod;
    /**
     * The list of webhook event triggers that are enabled for this Service: `onMessageAdded`, `onMessageUpdated`, `onMessageRemoved`, `onConversationUpdated`, `onConversationRemoved`, `onParticipantAdded`, `onParticipantUpdated`, `onParticipantRemoved`
     */
    filters: Array<string>;
    /**
     * The absolute url the pre-event webhook request should be sent to.
     */
    preWebhookUrl: string;
    /**
     * The absolute url the post-event webhook request should be sent to.
     */
    postWebhookUrl: string;
    target: WebhookTarget;
    /**
     * An absolute API resource API resource URL for this webhook.
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
        method: WebhookMethod;
        filters: string[];
        preWebhookUrl: string;
        postWebhookUrl: string;
        target: WebhookTarget;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WebhookSolution {
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
export declare function WebhookListInstance(version: V1): WebhookListInstance;
export {};
