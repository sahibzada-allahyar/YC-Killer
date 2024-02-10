/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
import { UserBindingListInstance } from "./user/userBinding";
import { UserChannelListInstance } from "./user/userChannel";
export type UserWebhookEnabledType = "true" | "false";
/**
 * Options to pass to update a UserInstance
 */
export interface UserContextUpdateOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: UserWebhookEnabledType;
    /** The SID of the [Role](https://www.twilio.com/docs/chat/rest/role-resource) to assign to the User. */
    roleSid?: string;
    /** A valid JSON string that contains application-specific data. */
    attributes?: string;
    /** A descriptive string that you create to describe the resource. It is often used for display purposes. */
    friendlyName?: string;
}
/**
 * Options to pass to create a UserInstance
 */
export interface UserListInstanceCreateOptions {
    /** The `identity` value that uniquely identifies the new resource\\\'s [User](https://www.twilio.com/docs/chat/rest/user-resource) within the [Service](https://www.twilio.com/docs/chat/rest/service-resource). This value is often a username or email address. See the Identity documentation for more info. */
    identity: string;
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: UserWebhookEnabledType;
    /** The SID of the [Role](https://www.twilio.com/docs/chat/rest/role-resource) to assign to the new User. */
    roleSid?: string;
    /** A valid JSON string that contains application-specific data. */
    attributes?: string;
    /** A descriptive string that you create to describe the new resource. This value is often used for display purposes. */
    friendlyName?: string;
}
/**
 * Options to pass to each
 */
export interface UserListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: UserInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface UserListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface UserListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface UserContext {
    userBindings: UserBindingListInstance;
    userChannels: UserChannelListInstance;
    /**
     * Remove a UserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a UserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    fetch(callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Update a UserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    update(callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Update a UserInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    update(params: UserContextUpdateOptions, callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface UserContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class UserContextImpl implements UserContext {
    protected _version: V2;
    protected _solution: UserContextSolution;
    protected _uri: string;
    protected _userBindings?: UserBindingListInstance;
    protected _userChannels?: UserChannelListInstance;
    constructor(_version: V2, serviceSid: string, sid: string);
    get userBindings(): UserBindingListInstance;
    get userChannels(): UserChannelListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    update(params?: UserContextUpdateOptions | ((error: Error | null, item?: UserInstance) => any), callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): UserContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface UserPayload extends TwilioResponsePayload {
    users: UserResource[];
}
interface UserResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    attributes: string;
    friendly_name: string;
    role_sid: string;
    identity: string;
    is_online: boolean;
    is_notifiable: boolean;
    date_created: Date;
    date_updated: Date;
    joined_channels_count: number;
    links: Record<string, string>;
    url: string;
}
export declare class UserInstance {
    protected _version: V2;
    protected _solution: UserContextSolution;
    protected _context?: UserContext;
    constructor(_version: V2, payload: UserResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the User resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the User resource.
     */
    accountSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/chat/rest/service-resource) the User resource is associated with.
     */
    serviceSid: string;
    /**
     * The JSON string that stores application-specific data. If attributes have not been set, `{}` is returned.
     */
    attributes: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The SID of the [Role](https://www.twilio.com/docs/chat/rest/role-resource) assigned to the user.
     */
    roleSid: string;
    /**
     * The application-defined string that uniquely identifies the resource\'s User within the [Service](https://www.twilio.com/docs/chat/rest/service-resource). This value is often a username or an email address, and is case-sensitive. See [access tokens](https://www.twilio.com/docs/chat/create-tokens) for more info.
     */
    identity: string;
    /**
     * Whether the User is actively connected to the Service instance and online. This value is only returned by Fetch actions that return a single resource and `null` is always returned by a Read action. This value is `null` if the Service\'s `reachability_enabled` is `false`, if the User has never been online for the Service instance, even if the Service\'s `reachability_enabled` is `true`.
     */
    isOnline: boolean;
    /**
     * Whether the User has a potentially valid Push Notification registration (APN or GCM) for the Service instance. If at least one registration exists, `true`; otherwise `false`. This value is only returned by Fetch actions that return a single resource and `null` is always returned by a Read action. This value is `null` if the Service\'s `reachability_enabled` is `false`, and if the User has never had a notification registration, even if the Service\'s `reachability_enabled` is `true`.
     */
    isNotifiable: boolean;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The number of Channels the User is a Member of.
     */
    joinedChannelsCount: number;
    /**
     * The absolute URLs of the [Channel](https://www.twilio.com/docs/chat/channels) and [Binding](https://www.twilio.com/docs/chat/rest/binding-resource) resources related to the user.
     */
    links: Record<string, string>;
    /**
     * The absolute URL of the User resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a UserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a UserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    fetch(callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Update a UserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    update(callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Update a UserInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    update(params: UserContextUpdateOptions, callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Access the userBindings.
     */
    userBindings(): UserBindingListInstance;
    /**
     * Access the userChannels.
     */
    userChannels(): UserChannelListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        attributes: string;
        friendlyName: string;
        roleSid: string;
        identity: string;
        isOnline: boolean;
        isNotifiable: boolean;
        dateCreated: Date;
        dateUpdated: Date;
        joinedChannelsCount: number;
        links: Record<string, string>;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface UserSolution {
    serviceSid: string;
}
export interface UserListInstance {
    _version: V2;
    _solution: UserSolution;
    _uri: string;
    (sid: string): UserContext;
    get(sid: string): UserContext;
    /**
     * Create a UserInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserInstance
     */
    create(params: UserListInstanceCreateOptions, callback?: (error: Error | null, item?: UserInstance) => any): Promise<UserInstance>;
    /**
     * Streams UserInstance records from the API.
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
     * @param { UserListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: UserInstance, done: (err?: Error) => void) => void): void;
    each(params: UserListInstanceEachOptions, callback?: (item: UserInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of UserInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: UserPage) => any): Promise<UserPage>;
    /**
     * Lists UserInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UserListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: UserInstance[]) => any): Promise<UserInstance[]>;
    list(params: UserListInstanceOptions, callback?: (error: Error | null, items: UserInstance[]) => any): Promise<UserInstance[]>;
    /**
     * Retrieve a single page of UserInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UserListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: UserPage) => any): Promise<UserPage>;
    page(params: UserListInstancePageOptions, callback?: (error: Error | null, items: UserPage) => any): Promise<UserPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function UserListInstance(version: V2, serviceSid: string): UserListInstance;
export declare class UserPage extends Page<V2, UserPayload, UserResource, UserInstance> {
    /**
     * Initialize the UserPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: UserSolution);
    /**
     * Build an instance of UserInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: UserResource): UserInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
