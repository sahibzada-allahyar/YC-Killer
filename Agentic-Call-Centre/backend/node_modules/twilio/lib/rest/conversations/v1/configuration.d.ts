/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
import { WebhookListInstance } from "./configuration/webhook";
/**
 * Options to pass to update a ConfigurationInstance
 */
export interface ConfigurationContextUpdateOptions {
    /** The SID of the default [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource) to use when creating a conversation. */
    defaultChatServiceSid?: string;
    /** The SID of the default [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) to use when creating a conversation. */
    defaultMessagingServiceSid?: string;
    /** Default ISO8601 duration when conversation will be switched to `inactive` state. Minimum value for this timer is 1 minute. */
    defaultInactiveTimer?: string;
    /** Default ISO8601 duration when conversation will be switched to `closed` state. Minimum value for this timer is 10 minutes. */
    defaultClosedTimer?: string;
}
export interface ConfigurationContext {
    /**
     * Fetch a ConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConfigurationInstance
     */
    fetch(callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    /**
     * Update a ConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConfigurationInstance
     */
    update(callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    /**
     * Update a ConfigurationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConfigurationInstance
     */
    update(params: ConfigurationContextUpdateOptions, callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ConfigurationContextSolution {
}
export declare class ConfigurationContextImpl implements ConfigurationContext {
    protected _version: V1;
    protected _solution: ConfigurationContextSolution;
    protected _uri: string;
    constructor(_version: V1);
    fetch(callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    update(params?: ConfigurationContextUpdateOptions | ((error: Error | null, item?: ConfigurationInstance) => any), callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ConfigurationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ConfigurationResource {
    account_sid: string;
    default_chat_service_sid: string;
    default_messaging_service_sid: string;
    default_inactive_timer: string;
    default_closed_timer: string;
    url: string;
    links: Record<string, string>;
}
export declare class ConfigurationInstance {
    protected _version: V1;
    protected _solution: ConfigurationContextSolution;
    protected _context?: ConfigurationContext;
    constructor(_version: V1, payload: ConfigurationResource);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this configuration.
     */
    accountSid: string;
    /**
     * The SID of the default [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource) used when creating a conversation.
     */
    defaultChatServiceSid: string;
    /**
     * The SID of the default [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) used when creating a conversation.
     */
    defaultMessagingServiceSid: string;
    /**
     * Default ISO8601 duration when conversation will be switched to `inactive` state. Minimum value for this timer is 1 minute.
     */
    defaultInactiveTimer: string;
    /**
     * Default ISO8601 duration when conversation will be switched to `closed` state. Minimum value for this timer is 10 minutes.
     */
    defaultClosedTimer: string;
    /**
     * An absolute API resource URL for this global configuration.
     */
    url: string;
    /**
     * Contains absolute API resource URLs to access the webhook and default service configurations.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a ConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConfigurationInstance
     */
    fetch(callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    /**
     * Update a ConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConfigurationInstance
     */
    update(callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    /**
     * Update a ConfigurationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConfigurationInstance
     */
    update(params: ConfigurationContextUpdateOptions, callback?: (error: Error | null, item?: ConfigurationInstance) => any): Promise<ConfigurationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        defaultChatServiceSid: string;
        defaultMessagingServiceSid: string;
        defaultInactiveTimer: string;
        defaultClosedTimer: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ConfigurationSolution {
}
export interface ConfigurationListInstance {
    _version: V1;
    _solution: ConfigurationSolution;
    _uri: string;
    (): ConfigurationContext;
    get(): ConfigurationContext;
    _webhooks?: WebhookListInstance;
    webhooks: WebhookListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ConfigurationListInstance(version: V1): ConfigurationListInstance;
export {};
