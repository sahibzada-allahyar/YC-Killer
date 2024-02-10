/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { ChannelListInstance } from "./service/channel";
import { RoleListInstance } from "./service/role";
import { UserListInstance } from "./service/user";
/**
 * Options to pass to update a ServiceInstance
 */
export interface ServiceContextUpdateOptions {
    /**  */
    friendlyName?: string;
    /**  */
    defaultServiceRoleSid?: string;
    /**  */
    defaultChannelRoleSid?: string;
    /**  */
    defaultChannelCreatorRoleSid?: string;
    /**  */
    readStatusEnabled?: boolean;
    /**  */
    reachabilityEnabled?: boolean;
    /**  */
    typingIndicatorTimeout?: number;
    /**  */
    consumptionReportInterval?: number;
    /**  */
    "notifications.newMessage.enabled"?: boolean;
    /**  */
    "notifications.newMessage.template"?: string;
    /**  */
    "notifications.addedToChannel.enabled"?: boolean;
    /**  */
    "notifications.addedToChannel.template"?: string;
    /**  */
    "notifications.removedFromChannel.enabled"?: boolean;
    /**  */
    "notifications.removedFromChannel.template"?: string;
    /**  */
    "notifications.invitedToChannel.enabled"?: boolean;
    /**  */
    "notifications.invitedToChannel.template"?: string;
    /**  */
    preWebhookUrl?: string;
    /**  */
    postWebhookUrl?: string;
    /**  */
    webhookMethod?: string;
    /**  */
    webhookFilters?: Array<string>;
    /**  */
    "webhooks.onMessageSend.url"?: string;
    /**  */
    "webhooks.onMessageSend.method"?: string;
    /**  */
    "webhooks.onMessageUpdate.url"?: string;
    /**  */
    "webhooks.onMessageUpdate.method"?: string;
    /**  */
    "webhooks.onMessageRemove.url"?: string;
    /**  */
    "webhooks.onMessageRemove.method"?: string;
    /**  */
    "webhooks.onChannelAdd.url"?: string;
    /**  */
    "webhooks.onChannelAdd.method"?: string;
    /**  */
    "webhooks.onChannelDestroy.url"?: string;
    /**  */
    "webhooks.onChannelDestroy.method"?: string;
    /**  */
    "webhooks.onChannelUpdate.url"?: string;
    /**  */
    "webhooks.onChannelUpdate.method"?: string;
    /**  */
    "webhooks.onMemberAdd.url"?: string;
    /**  */
    "webhooks.onMemberAdd.method"?: string;
    /**  */
    "webhooks.onMemberRemove.url"?: string;
    /**  */
    "webhooks.onMemberRemove.method"?: string;
    /**  */
    "webhooks.onMessageSent.url"?: string;
    /**  */
    "webhooks.onMessageSent.method"?: string;
    /**  */
    "webhooks.onMessageUpdated.url"?: string;
    /**  */
    "webhooks.onMessageUpdated.method"?: string;
    /**  */
    "webhooks.onMessageRemoved.url"?: string;
    /**  */
    "webhooks.onMessageRemoved.method"?: string;
    /**  */
    "webhooks.onChannelAdded.url"?: string;
    /**  */
    "webhooks.onChannelAdded.method"?: string;
    /**  */
    "webhooks.onChannelDestroyed.url"?: string;
    /**  */
    "webhooks.onChannelDestroyed.method"?: string;
    /**  */
    "webhooks.onChannelUpdated.url"?: string;
    /**  */
    "webhooks.onChannelUpdated.method"?: string;
    /**  */
    "webhooks.onMemberAdded.url"?: string;
    /**  */
    "webhooks.onMemberAdded.method"?: string;
    /**  */
    "webhooks.onMemberRemoved.url"?: string;
    /**  */
    "webhooks.onMemberRemoved.method"?: string;
    /**  */
    "limits.channelMembers"?: number;
    /**  */
    "limits.userChannels"?: number;
}
/**
 * Options to pass to create a ServiceInstance
 */
export interface ServiceListInstanceCreateOptions {
    /**  */
    friendlyName: string;
}
/**
 * Options to pass to each
 */
export interface ServiceListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ServiceInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ServiceListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ServiceListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ServiceContext {
    channels: ChannelListInstance;
    roles: RoleListInstance;
    users: UserListInstance;
    /**
     * Remove a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    fetch(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Update a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    update(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Update a ServiceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    update(params: ServiceContextUpdateOptions, callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ServiceContextSolution {
    sid: string;
}
export declare class ServiceContextImpl implements ServiceContext {
    protected _version: V1;
    protected _solution: ServiceContextSolution;
    protected _uri: string;
    protected _channels?: ChannelListInstance;
    protected _roles?: RoleListInstance;
    protected _users?: UserListInstance;
    constructor(_version: V1, sid: string);
    get channels(): ChannelListInstance;
    get roles(): RoleListInstance;
    get users(): UserListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    update(params?: ServiceContextUpdateOptions | ((error: Error | null, item?: ServiceInstance) => any), callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ServiceContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ServicePayload extends TwilioResponsePayload {
    services: ServiceResource[];
}
interface ServiceResource {
    sid: string;
    account_sid: string;
    friendly_name: string;
    date_created: Date;
    date_updated: Date;
    default_service_role_sid: string;
    default_channel_role_sid: string;
    default_channel_creator_role_sid: string;
    read_status_enabled: boolean;
    reachability_enabled: boolean;
    typing_indicator_timeout: number;
    consumption_report_interval: number;
    limits: any;
    webhooks: any;
    pre_webhook_url: string;
    post_webhook_url: string;
    webhook_method: string;
    webhook_filters: Array<string>;
    notifications: any;
    url: string;
    links: Record<string, string>;
}
export declare class ServiceInstance {
    protected _version: V1;
    protected _solution: ServiceContextSolution;
    protected _context?: ServiceContext;
    constructor(_version: V1, payload: ServiceResource, sid?: string);
    sid: string;
    accountSid: string;
    friendlyName: string;
    dateCreated: Date;
    dateUpdated: Date;
    defaultServiceRoleSid: string;
    defaultChannelRoleSid: string;
    defaultChannelCreatorRoleSid: string;
    readStatusEnabled: boolean;
    reachabilityEnabled: boolean;
    typingIndicatorTimeout: number;
    consumptionReportInterval: number;
    limits: any;
    webhooks: any;
    preWebhookUrl: string;
    postWebhookUrl: string;
    webhookMethod: string;
    webhookFilters: Array<string>;
    notifications: any;
    url: string;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    fetch(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Update a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    update(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Update a ServiceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    update(params: ServiceContextUpdateOptions, callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Access the channels.
     */
    channels(): ChannelListInstance;
    /**
     * Access the roles.
     */
    roles(): RoleListInstance;
    /**
     * Access the users.
     */
    users(): UserListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        friendlyName: string;
        dateCreated: Date;
        dateUpdated: Date;
        defaultServiceRoleSid: string;
        defaultChannelRoleSid: string;
        defaultChannelCreatorRoleSid: string;
        readStatusEnabled: boolean;
        reachabilityEnabled: boolean;
        typingIndicatorTimeout: number;
        consumptionReportInterval: number;
        limits: any;
        webhooks: any;
        preWebhookUrl: string;
        postWebhookUrl: string;
        webhookMethod: string;
        webhookFilters: string[];
        notifications: any;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ServiceSolution {
}
export interface ServiceListInstance {
    _version: V1;
    _solution: ServiceSolution;
    _uri: string;
    (sid: string): ServiceContext;
    get(sid: string): ServiceContext;
    /**
     * Create a ServiceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    create(params: ServiceListInstanceCreateOptions, callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Streams ServiceInstance records from the API.
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
     * @param { ServiceListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ServiceInstance, done: (err?: Error) => void) => void): void;
    each(params: ServiceListInstanceEachOptions, callback?: (item: ServiceInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ServiceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ServicePage) => any): Promise<ServicePage>;
    /**
     * Lists ServiceInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ServiceListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ServiceInstance[]) => any): Promise<ServiceInstance[]>;
    list(params: ServiceListInstanceOptions, callback?: (error: Error | null, items: ServiceInstance[]) => any): Promise<ServiceInstance[]>;
    /**
     * Retrieve a single page of ServiceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ServiceListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ServicePage) => any): Promise<ServicePage>;
    page(params: ServiceListInstancePageOptions, callback?: (error: Error | null, items: ServicePage) => any): Promise<ServicePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ServiceListInstance(version: V1): ServiceListInstance;
export declare class ServicePage extends Page<V1, ServicePayload, ServiceResource, ServiceInstance> {
    /**
     * Initialize the ServicePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ServiceSolution);
    /**
     * Build an instance of ServiceInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ServiceResource): ServiceInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
