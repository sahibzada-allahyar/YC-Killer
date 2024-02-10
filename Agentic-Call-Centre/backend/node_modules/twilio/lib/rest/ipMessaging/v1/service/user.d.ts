/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { UserChannelListInstance } from "./user/userChannel";
/**
 * Options to pass to update a UserInstance
 */
export interface UserContextUpdateOptions {
    /**  */
    roleSid?: string;
    /**  */
    attributes?: string;
    /**  */
    friendlyName?: string;
}
/**
 * Options to pass to create a UserInstance
 */
export interface UserListInstanceCreateOptions {
    /**  */
    identity: string;
    /**  */
    roleSid?: string;
    /**  */
    attributes?: string;
    /**  */
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
    protected _version: V1;
    protected _solution: UserContextSolution;
    protected _uri: string;
    protected _userChannels?: UserChannelListInstance;
    constructor(_version: V1, serviceSid: string, sid: string);
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
    protected _version: V1;
    protected _solution: UserContextSolution;
    protected _context?: UserContext;
    constructor(_version: V1, payload: UserResource, serviceSid: string, sid?: string);
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
    _version: V1;
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
export declare function UserListInstance(version: V1, serviceSid: string): UserListInstance;
export declare class UserPage extends Page<V1, UserPayload, UserResource, UserInstance> {
    /**
     * Initialize the UserPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: UserSolution);
    /**
     * Build an instance of UserInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: UserResource): UserInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
