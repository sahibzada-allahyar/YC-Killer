/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
/**
 * Options to pass to update a NotificationInstance
 */
export interface NotificationContextUpdateOptions {
    /** Weather the notification logging is enabled. */
    logEnabled?: boolean;
    /** Whether to send a notification when a new message is added to a conversation. The default is `false`. */
    "newMessage.enabled"?: boolean;
    /** The template to use to create the notification text displayed when a new message is added to a conversation and `new_message.enabled` is `true`. */
    "newMessage.template"?: string;
    /** The name of the sound to play when a new message is added to a conversation and `new_message.enabled` is `true`. */
    "newMessage.sound"?: string;
    /** Whether the new message badge is enabled. The default is `false`. */
    "newMessage.badgeCountEnabled"?: boolean;
    /** Whether to send a notification when a participant is added to a conversation. The default is `false`. */
    "addedToConversation.enabled"?: boolean;
    /** The template to use to create the notification text displayed when a participant is added to a conversation and `added_to_conversation.enabled` is `true`. */
    "addedToConversation.template"?: string;
    /** The name of the sound to play when a participant is added to a conversation and `added_to_conversation.enabled` is `true`. */
    "addedToConversation.sound"?: string;
    /** Whether to send a notification to a user when they are removed from a conversation. The default is `false`. */
    "removedFromConversation.enabled"?: boolean;
    /** The template to use to create the notification text displayed to a user when they are removed from a conversation and `removed_from_conversation.enabled` is `true`. */
    "removedFromConversation.template"?: string;
    /** The name of the sound to play to a user when they are removed from a conversation and `removed_from_conversation.enabled` is `true`. */
    "removedFromConversation.sound"?: string;
    /** Whether to send a notification when a new message with media/file attachments is added to a conversation. The default is `false`. */
    "newMessage.withMedia.enabled"?: boolean;
    /** The template to use to create the notification text displayed when a new message with media/file attachments is added to a conversation and `new_message.attachments.enabled` is `true`. */
    "newMessage.withMedia.template"?: string;
}
export interface NotificationContext {
    /**
     * Fetch a NotificationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NotificationInstance
     */
    fetch(callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    /**
     * Update a NotificationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NotificationInstance
     */
    update(callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    /**
     * Update a NotificationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NotificationInstance
     */
    update(params: NotificationContextUpdateOptions, callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface NotificationContextSolution {
    chatServiceSid: string;
}
export declare class NotificationContextImpl implements NotificationContext {
    protected _version: V1;
    protected _solution: NotificationContextSolution;
    protected _uri: string;
    constructor(_version: V1, chatServiceSid: string);
    fetch(callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    update(params?: NotificationContextUpdateOptions | ((error: Error | null, item?: NotificationInstance) => any), callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): NotificationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface NotificationResource {
    account_sid: string;
    chat_service_sid: string;
    new_message: any;
    added_to_conversation: any;
    removed_from_conversation: any;
    log_enabled: boolean;
    url: string;
}
export declare class NotificationInstance {
    protected _version: V1;
    protected _solution: NotificationContextSolution;
    protected _context?: NotificationContext;
    constructor(_version: V1, payload: NotificationResource, chatServiceSid: string);
    /**
     * The unique ID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this configuration.
     */
    accountSid: string;
    /**
     * The SID of the [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource) the Configuration applies to.
     */
    chatServiceSid: string;
    /**
     * The Push Notification configuration for New Messages.
     */
    newMessage: any;
    /**
     * The Push Notification configuration for being added to a Conversation.
     */
    addedToConversation: any;
    /**
     * The Push Notification configuration for being removed from a Conversation.
     */
    removedFromConversation: any;
    /**
     * Weather the notification logging is enabled.
     */
    logEnabled: boolean;
    /**
     * An absolute API resource URL for this configuration.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a NotificationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NotificationInstance
     */
    fetch(callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    /**
     * Update a NotificationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NotificationInstance
     */
    update(callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    /**
     * Update a NotificationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NotificationInstance
     */
    update(params: NotificationContextUpdateOptions, callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        chatServiceSid: string;
        newMessage: any;
        addedToConversation: any;
        removedFromConversation: any;
        logEnabled: boolean;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface NotificationSolution {
    chatServiceSid: string;
}
export interface NotificationListInstance {
    _version: V1;
    _solution: NotificationSolution;
    _uri: string;
    (): NotificationContext;
    get(): NotificationContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function NotificationListInstance(version: V1, chatServiceSid: string): NotificationListInstance;
export {};
