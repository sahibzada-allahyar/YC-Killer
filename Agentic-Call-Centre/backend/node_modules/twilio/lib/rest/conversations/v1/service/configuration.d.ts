/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
import { NotificationListInstance } from "./configuration/notification";
import { WebhookListInstance } from "./configuration/webhook";
/**
 * Options to pass to update a ConfigurationInstance
 */
export interface ConfigurationContextUpdateOptions {
    /** The conversation-level role assigned to a conversation creator when they join a new conversation. See [Conversation Role](https://www.twilio.com/docs/conversations/api/role-resource) for more info about roles. */
    defaultConversationCreatorRoleSid?: string;
    /** The conversation-level role assigned to users when they are added to a conversation. See [Conversation Role](https://www.twilio.com/docs/conversations/api/role-resource) for more info about roles. */
    defaultConversationRoleSid?: string;
    /** The service-level role assigned to users when they are added to the service. See [Conversation Role](https://www.twilio.com/docs/conversations/api/role-resource) for more info about roles. */
    defaultChatServiceRoleSid?: string;
    /** Whether the [Reachability Indicator](https://www.twilio.com/docs/conversations/reachability) is enabled for this Conversations Service. The default is `false`. */
    reachabilityEnabled?: boolean;
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
    chatServiceSid: string;
}
export declare class ConfigurationContextImpl implements ConfigurationContext {
    protected _version: V1;
    protected _solution: ConfigurationContextSolution;
    protected _uri: string;
    constructor(_version: V1, chatServiceSid: string);
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
    chat_service_sid: string;
    default_conversation_creator_role_sid: string;
    default_conversation_role_sid: string;
    default_chat_service_role_sid: string;
    url: string;
    links: Record<string, string>;
    reachability_enabled: boolean;
}
export declare class ConfigurationInstance {
    protected _version: V1;
    protected _solution: ConfigurationContextSolution;
    protected _context?: ConfigurationContext;
    constructor(_version: V1, payload: ConfigurationResource, chatServiceSid: string);
    /**
     * The unique string that we created to identify the Service configuration resource.
     */
    chatServiceSid: string;
    /**
     * The conversation-level role assigned to a conversation creator when they join a new conversation. See [Conversation Role](https://www.twilio.com/docs/conversations/api/role-resource) for more info about roles.
     */
    defaultConversationCreatorRoleSid: string;
    /**
     * The conversation-level role assigned to users when they are added to a conversation. See [Conversation Role](https://www.twilio.com/docs/conversations/api/role-resource) for more info about roles.
     */
    defaultConversationRoleSid: string;
    /**
     * The service-level role assigned to users when they are added to the service. See [Conversation Role](https://www.twilio.com/docs/conversations/api/role-resource) for more info about roles.
     */
    defaultChatServiceRoleSid: string;
    /**
     * An absolute API resource URL for this service configuration.
     */
    url: string;
    /**
     * Contains an absolute API resource URL to access the push notifications configuration of this service.
     */
    links: Record<string, string>;
    /**
     * Whether the [Reachability Indicator](https://www.twilio.com/docs/conversations/reachability) is enabled for this Conversations Service. The default is `false`.
     */
    reachabilityEnabled: boolean;
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
        chatServiceSid: string;
        defaultConversationCreatorRoleSid: string;
        defaultConversationRoleSid: string;
        defaultChatServiceRoleSid: string;
        url: string;
        links: Record<string, string>;
        reachabilityEnabled: boolean;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ConfigurationSolution {
    chatServiceSid: string;
}
export interface ConfigurationListInstance {
    _version: V1;
    _solution: ConfigurationSolution;
    _uri: string;
    (): ConfigurationContext;
    get(): ConfigurationContext;
    _notifications?: NotificationListInstance;
    notifications: NotificationListInstance;
    _webhooks?: WebhookListInstance;
    webhooks: WebhookListInstance;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ConfigurationListInstance(version: V1, chatServiceSid: string): ConfigurationListInstance;
export {};
