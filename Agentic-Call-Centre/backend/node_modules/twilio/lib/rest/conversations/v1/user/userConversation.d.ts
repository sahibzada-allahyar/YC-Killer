/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type UserConversationNotificationLevel = "default" | "muted";
export type UserConversationState = "inactive" | "active" | "closed";
/**
 * Options to pass to update a UserConversationInstance
 */
export interface UserConversationContextUpdateOptions {
    /**  */
    notificationLevel?: UserConversationNotificationLevel;
    /** The date of the last message read in conversation by the user, given in ISO 8601 format. */
    lastReadTimestamp?: Date;
    /** The index of the last Message in the Conversation that the Participant has read. */
    lastReadMessageIndex?: number;
}
/**
 * Options to pass to each
 */
export interface UserConversationListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: UserConversationInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface UserConversationListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface UserConversationListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface UserConversationContext {
    /**
     * Remove a UserConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a UserConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserConversationInstance
     */
    fetch(callback?: (error: Error | null, item?: UserConversationInstance) => any): Promise<UserConversationInstance>;
    /**
     * Update a UserConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserConversationInstance
     */
    update(callback?: (error: Error | null, item?: UserConversationInstance) => any): Promise<UserConversationInstance>;
    /**
     * Update a UserConversationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserConversationInstance
     */
    update(params: UserConversationContextUpdateOptions, callback?: (error: Error | null, item?: UserConversationInstance) => any): Promise<UserConversationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface UserConversationContextSolution {
    userSid: string;
    conversationSid: string;
}
export declare class UserConversationContextImpl implements UserConversationContext {
    protected _version: V1;
    protected _solution: UserConversationContextSolution;
    protected _uri: string;
    constructor(_version: V1, userSid: string, conversationSid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: UserConversationInstance) => any): Promise<UserConversationInstance>;
    update(params?: UserConversationContextUpdateOptions | ((error: Error | null, item?: UserConversationInstance) => any), callback?: (error: Error | null, item?: UserConversationInstance) => any): Promise<UserConversationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): UserConversationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface UserConversationPayload extends TwilioResponsePayload {
    conversations: UserConversationResource[];
}
interface UserConversationResource {
    account_sid: string;
    chat_service_sid: string;
    conversation_sid: string;
    unread_messages_count: number;
    last_read_message_index: number;
    participant_sid: string;
    user_sid: string;
    friendly_name: string;
    conversation_state: UserConversationState;
    timers: any;
    attributes: string;
    date_created: Date;
    date_updated: Date;
    created_by: string;
    notification_level: UserConversationNotificationLevel;
    unique_name: string;
    url: string;
    links: Record<string, string>;
}
export declare class UserConversationInstance {
    protected _version: V1;
    protected _solution: UserConversationContextSolution;
    protected _context?: UserConversationContext;
    constructor(_version: V1, payload: UserConversationResource, userSid: string, conversationSid?: string);
    /**
     * The unique ID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this conversation.
     */
    accountSid: string;
    /**
     * The unique ID of the [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource) this conversation belongs to.
     */
    chatServiceSid: string;
    /**
     * The unique ID of the [Conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) for this User Conversation.
     */
    conversationSid: string;
    /**
     * The number of unread Messages in the Conversation for the Participant.
     */
    unreadMessagesCount: number;
    /**
     * The index of the last Message in the Conversation that the Participant has read.
     */
    lastReadMessageIndex: number;
    /**
     * The unique ID of the [participant](https://www.twilio.com/docs/conversations/api/conversation-participant-resource) the user conversation belongs to.
     */
    participantSid: string;
    /**
     * The unique string that identifies the [User resource](https://www.twilio.com/docs/conversations/api/user-resource).
     */
    userSid: string;
    /**
     * The human-readable name of this conversation, limited to 256 characters. Optional.
     */
    friendlyName: string;
    conversationState: UserConversationState;
    /**
     * Timer date values representing state update for this conversation.
     */
    timers: any;
    /**
     * An optional string metadata field you can use to store any data you wish. The string value must contain structurally valid JSON if specified.  **Note** that if the attributes are not set \"{}\" will be returned.
     */
    attributes: string;
    /**
     * The date that this conversation was created, given in ISO 8601 format.
     */
    dateCreated: Date;
    /**
     * The date that this conversation was last updated, given in ISO 8601 format.
     */
    dateUpdated: Date;
    /**
     * Identity of the creator of this Conversation.
     */
    createdBy: string;
    notificationLevel: UserConversationNotificationLevel;
    /**
     * An application-defined string that uniquely identifies the Conversation resource. It can be used to address the resource in place of the resource\'s `conversation_sid` in the URL.
     */
    uniqueName: string;
    url: string;
    /**
     * Contains absolute URLs to access the [participant](https://www.twilio.com/docs/conversations/api/conversation-participant-resource) and [conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) of this conversation.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a UserConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a UserConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserConversationInstance
     */
    fetch(callback?: (error: Error | null, item?: UserConversationInstance) => any): Promise<UserConversationInstance>;
    /**
     * Update a UserConversationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserConversationInstance
     */
    update(callback?: (error: Error | null, item?: UserConversationInstance) => any): Promise<UserConversationInstance>;
    /**
     * Update a UserConversationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserConversationInstance
     */
    update(params: UserConversationContextUpdateOptions, callback?: (error: Error | null, item?: UserConversationInstance) => any): Promise<UserConversationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        chatServiceSid: string;
        conversationSid: string;
        unreadMessagesCount: number;
        lastReadMessageIndex: number;
        participantSid: string;
        userSid: string;
        friendlyName: string;
        conversationState: UserConversationState;
        timers: any;
        attributes: string;
        dateCreated: Date;
        dateUpdated: Date;
        createdBy: string;
        notificationLevel: UserConversationNotificationLevel;
        uniqueName: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface UserConversationSolution {
    userSid: string;
}
export interface UserConversationListInstance {
    _version: V1;
    _solution: UserConversationSolution;
    _uri: string;
    (conversationSid: string): UserConversationContext;
    get(conversationSid: string): UserConversationContext;
    /**
     * Streams UserConversationInstance records from the API.
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
     * @param { UserConversationListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: UserConversationInstance, done: (err?: Error) => void) => void): void;
    each(params: UserConversationListInstanceEachOptions, callback?: (item: UserConversationInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of UserConversationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: UserConversationPage) => any): Promise<UserConversationPage>;
    /**
     * Lists UserConversationInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UserConversationListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: UserConversationInstance[]) => any): Promise<UserConversationInstance[]>;
    list(params: UserConversationListInstanceOptions, callback?: (error: Error | null, items: UserConversationInstance[]) => any): Promise<UserConversationInstance[]>;
    /**
     * Retrieve a single page of UserConversationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UserConversationListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: UserConversationPage) => any): Promise<UserConversationPage>;
    page(params: UserConversationListInstancePageOptions, callback?: (error: Error | null, items: UserConversationPage) => any): Promise<UserConversationPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function UserConversationListInstance(version: V1, userSid: string): UserConversationListInstance;
export declare class UserConversationPage extends Page<V1, UserConversationPayload, UserConversationResource, UserConversationInstance> {
    /**
     * Initialize the UserConversationPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: UserConversationSolution);
    /**
     * Build an instance of UserConversationInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: UserConversationResource): UserConversationInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
