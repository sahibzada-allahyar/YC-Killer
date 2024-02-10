/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2 from "../../../V2";
export type UserChannelChannelStatus = "joined" | "invited" | "not_participating";
export type UserChannelNotificationLevel = "default" | "muted";
export type UserChannelWebhookEnabledType = "true" | "false";
/**
 * Options to pass to remove a UserChannelInstance
 */
export interface UserChannelContextRemoveOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: UserChannelWebhookEnabledType;
}
/**
 * Options to pass to update a UserChannelInstance
 */
export interface UserChannelContextUpdateOptions {
    /**  */
    notificationLevel?: UserChannelNotificationLevel;
    /** The index of the last [Message](https://www.twilio.com/docs/chat/rest/message-resource) in the [Channel](https://www.twilio.com/docs/chat/channels) that the Member has read. */
    lastConsumedMessageIndex?: number;
    /** The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) timestamp of the last [Message](https://www.twilio.com/docs/chat/rest/message-resource) read event for the Member within the [Channel](https://www.twilio.com/docs/chat/channels). */
    lastConsumptionTimestamp?: Date;
}
/**
 * Options to pass to each
 */
export interface UserChannelListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: UserChannelInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface UserChannelListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface UserChannelListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface UserChannelContext {
    /**
     * Remove a UserChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a UserChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserChannelInstance
     */
    remove(params: UserChannelContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a UserChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: UserChannelInstance) => any): Promise<UserChannelInstance>;
    /**
     * Update a UserChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserChannelInstance
     */
    update(callback?: (error: Error | null, item?: UserChannelInstance) => any): Promise<UserChannelInstance>;
    /**
     * Update a UserChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserChannelInstance
     */
    update(params: UserChannelContextUpdateOptions, callback?: (error: Error | null, item?: UserChannelInstance) => any): Promise<UserChannelInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface UserChannelContextSolution {
    serviceSid: string;
    userSid: string;
    channelSid: string;
}
export declare class UserChannelContextImpl implements UserChannelContext {
    protected _version: V2;
    protected _solution: UserChannelContextSolution;
    protected _uri: string;
    constructor(_version: V2, serviceSid: string, userSid: string, channelSid: string);
    remove(params?: UserChannelContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: UserChannelInstance) => any): Promise<UserChannelInstance>;
    update(params?: UserChannelContextUpdateOptions | ((error: Error | null, item?: UserChannelInstance) => any), callback?: (error: Error | null, item?: UserChannelInstance) => any): Promise<UserChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): UserChannelContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface UserChannelPayload extends TwilioResponsePayload {
    channels: UserChannelResource[];
}
interface UserChannelResource {
    account_sid: string;
    service_sid: string;
    channel_sid: string;
    user_sid: string;
    member_sid: string;
    status: UserChannelChannelStatus;
    last_consumed_message_index: number;
    unread_messages_count: number;
    links: Record<string, string>;
    url: string;
    notification_level: UserChannelNotificationLevel;
}
export declare class UserChannelInstance {
    protected _version: V2;
    protected _solution: UserChannelContextSolution;
    protected _context?: UserChannelContext;
    constructor(_version: V2, payload: UserChannelResource, serviceSid: string, userSid: string, channelSid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the User Channel resource.
     */
    accountSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/chat/rest/service-resource) the User Channel resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the [Channel](https://www.twilio.com/docs/chat/channels) the User Channel resource belongs to.
     */
    channelSid: string;
    /**
     * The SID of the [User](https://www.twilio.com/docs/chat/rest/user-resource) the User Channel belongs to.
     */
    userSid: string;
    /**
     * The SID of a [Member](https://www.twilio.com/docs/chat/rest/member-resource) that represents the User on the Channel.
     */
    memberSid: string;
    status: UserChannelChannelStatus;
    /**
     * The index of the last [Message](https://www.twilio.com/docs/chat/rest/message-resource) in the [Channel](https://www.twilio.com/docs/chat/channels) that the Member has read.
     */
    lastConsumedMessageIndex: number;
    /**
     * The number of unread Messages in the Channel for the User. Note that retrieving messages on a client endpoint does not mean that messages are consumed or read. See [Consumption Horizon feature](https://www.twilio.com/docs/chat/consumption-horizon) to learn how to mark messages as consumed.
     */
    unreadMessagesCount: number;
    /**
     * The absolute URLs of the [Members](https://www.twilio.com/docs/chat/rest/member-resource), [Messages](https://www.twilio.com/docs/chat/rest/message-resource) , [Invites](https://www.twilio.com/docs/chat/rest/invite-resource) and, if it exists, the last [Message](https://www.twilio.com/docs/chat/rest/message-resource) for the Channel.
     */
    links: Record<string, string>;
    /**
     * The absolute URL of the User Channel resource.
     */
    url: string;
    notificationLevel: UserChannelNotificationLevel;
    private get _proxy();
    /**
     * Remove a UserChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a UserChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserChannelInstance
     */
    remove(params: UserChannelContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a UserChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: UserChannelInstance) => any): Promise<UserChannelInstance>;
    /**
     * Update a UserChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserChannelInstance
     */
    update(callback?: (error: Error | null, item?: UserChannelInstance) => any): Promise<UserChannelInstance>;
    /**
     * Update a UserChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserChannelInstance
     */
    update(params: UserChannelContextUpdateOptions, callback?: (error: Error | null, item?: UserChannelInstance) => any): Promise<UserChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        serviceSid: string;
        channelSid: string;
        userSid: string;
        memberSid: string;
        status: UserChannelChannelStatus;
        lastConsumedMessageIndex: number;
        unreadMessagesCount: number;
        links: Record<string, string>;
        url: string;
        notificationLevel: UserChannelNotificationLevel;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface UserChannelSolution {
    serviceSid: string;
    userSid: string;
}
export interface UserChannelListInstance {
    _version: V2;
    _solution: UserChannelSolution;
    _uri: string;
    (channelSid: string): UserChannelContext;
    get(channelSid: string): UserChannelContext;
    /**
     * Streams UserChannelInstance records from the API.
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
     * @param { UserChannelListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: UserChannelInstance, done: (err?: Error) => void) => void): void;
    each(params: UserChannelListInstanceEachOptions, callback?: (item: UserChannelInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of UserChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: UserChannelPage) => any): Promise<UserChannelPage>;
    /**
     * Lists UserChannelInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UserChannelListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: UserChannelInstance[]) => any): Promise<UserChannelInstance[]>;
    list(params: UserChannelListInstanceOptions, callback?: (error: Error | null, items: UserChannelInstance[]) => any): Promise<UserChannelInstance[]>;
    /**
     * Retrieve a single page of UserChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UserChannelListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: UserChannelPage) => any): Promise<UserChannelPage>;
    page(params: UserChannelListInstancePageOptions, callback?: (error: Error | null, items: UserChannelPage) => any): Promise<UserChannelPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function UserChannelListInstance(version: V2, serviceSid: string, userSid: string): UserChannelListInstance;
export declare class UserChannelPage extends Page<V2, UserChannelPayload, UserChannelResource, UserChannelInstance> {
    /**
     * Initialize the UserChannelPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: UserChannelSolution);
    /**
     * Build an instance of UserChannelInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: UserChannelResource): UserChannelInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
