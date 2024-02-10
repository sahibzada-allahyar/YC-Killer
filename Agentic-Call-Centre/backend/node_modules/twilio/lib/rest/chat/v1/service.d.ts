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
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** The service role assigned to users when they are added to the service. See the [Roles endpoint](https://www.twilio.com/docs/chat/api/roles) for more details. */
    defaultServiceRoleSid?: string;
    /** The channel role assigned to users when they are added to a channel. See the [Roles endpoint](https://www.twilio.com/docs/chat/api/roles) for more details. */
    defaultChannelRoleSid?: string;
    /** The channel role assigned to a channel creator when they join a new channel. See the [Roles endpoint](https://www.twilio.com/docs/chat/api/roles) for more details. */
    defaultChannelCreatorRoleSid?: string;
    /** Whether to enable the [Message Consumption Horizon](https://www.twilio.com/docs/chat/consumption-horizon) feature. The default is `true`. */
    readStatusEnabled?: boolean;
    /** Whether to enable the [Reachability Indicator](https://www.twilio.com/docs/chat/reachability-indicator) for this Service instance. The default is `false`. */
    reachabilityEnabled?: boolean;
    /** How long in seconds after a `started typing` event until clients should assume that user is no longer typing, even if no `ended typing` message was received.  The default is 5 seconds. */
    typingIndicatorTimeout?: number;
    /** DEPRECATED. The interval in seconds between consumption reports submission batches from client endpoints. */
    consumptionReportInterval?: number;
    /** Whether to send a notification when a new message is added to a channel. Can be: `true` or `false` and the default is `false`. */
    "notifications.newMessage.enabled"?: boolean;
    /** The template to use to create the notification text displayed when a new message is added to a channel and `notifications.new_message.enabled` is `true`. */
    "notifications.newMessage.template"?: string;
    /** Whether to send a notification when a member is added to a channel. Can be: `true` or `false` and the default is `false`. */
    "notifications.addedToChannel.enabled"?: boolean;
    /** The template to use to create the notification text displayed when a member is added to a channel and `notifications.added_to_channel.enabled` is `true`. */
    "notifications.addedToChannel.template"?: string;
    /** Whether to send a notification to a user when they are removed from a channel. Can be: `true` or `false` and the default is `false`. */
    "notifications.removedFromChannel.enabled"?: boolean;
    /** The template to use to create the notification text displayed to a user when they are removed from a channel and `notifications.removed_from_channel.enabled` is `true`. */
    "notifications.removedFromChannel.template"?: string;
    /** Whether to send a notification when a user is invited to a channel. Can be: `true` or `false` and the default is `false`. */
    "notifications.invitedToChannel.enabled"?: boolean;
    /** The template to use to create the notification text displayed when a user is invited to a channel and `notifications.invited_to_channel.enabled` is `true`. */
    "notifications.invitedToChannel.template"?: string;
    /** The URL for pre-event webhooks, which are called by using the `webhook_method`. See [Webhook Events](https://www.twilio.com/docs/api/chat/webhooks) for more details. */
    preWebhookUrl?: string;
    /** The URL for post-event webhooks, which are called by using the `webhook_method`. See [Webhook Events](https://www.twilio.com/docs/api/chat/webhooks) for more details. */
    postWebhookUrl?: string;
    /** The HTTP method to use for calls to the `pre_webhook_url` and `post_webhook_url` webhooks.  Can be: `POST` or `GET` and the default is `POST`. See [Webhook Events](https://www.twilio.com/docs/chat/webhook-events) for more details. */
    webhookMethod?: string;
    /** The list of WebHook events that are enabled for this Service instance. See [Webhook Events](https://www.twilio.com/docs/chat/webhook-events) for more details. */
    webhookFilters?: Array<string>;
    /** The URL of the webhook to call in response to the `on_message_send` event using the `webhooks.on_message_send.method` HTTP method. */
    "webhooks.onMessageSend.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_message_send.url`. */
    "webhooks.onMessageSend.method"?: string;
    /** The URL of the webhook to call in response to the `on_message_update` event using the `webhooks.on_message_update.method` HTTP method. */
    "webhooks.onMessageUpdate.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_message_update.url`. */
    "webhooks.onMessageUpdate.method"?: string;
    /** The URL of the webhook to call in response to the `on_message_remove` event using the `webhooks.on_message_remove.method` HTTP method. */
    "webhooks.onMessageRemove.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_message_remove.url`. */
    "webhooks.onMessageRemove.method"?: string;
    /** The URL of the webhook to call in response to the `on_channel_add` event using the `webhooks.on_channel_add.method` HTTP method. */
    "webhooks.onChannelAdd.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_channel_add.url`. */
    "webhooks.onChannelAdd.method"?: string;
    /** The URL of the webhook to call in response to the `on_channel_destroy` event using the `webhooks.on_channel_destroy.method` HTTP method. */
    "webhooks.onChannelDestroy.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_channel_destroy.url`. */
    "webhooks.onChannelDestroy.method"?: string;
    /** The URL of the webhook to call in response to the `on_channel_update` event using the `webhooks.on_channel_update.method` HTTP method. */
    "webhooks.onChannelUpdate.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_channel_update.url`. */
    "webhooks.onChannelUpdate.method"?: string;
    /** The URL of the webhook to call in response to the `on_member_add` event using the `webhooks.on_member_add.method` HTTP method. */
    "webhooks.onMemberAdd.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_member_add.url`. */
    "webhooks.onMemberAdd.method"?: string;
    /** The URL of the webhook to call in response to the `on_member_remove` event using the `webhooks.on_member_remove.method` HTTP method. */
    "webhooks.onMemberRemove.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_member_remove.url`. */
    "webhooks.onMemberRemove.method"?: string;
    /** The URL of the webhook to call in response to the `on_message_sent` event using the `webhooks.on_message_sent.method` HTTP method. */
    "webhooks.onMessageSent.url"?: string;
    /** The URL of the webhook to call in response to the `on_message_sent` event`. */
    "webhooks.onMessageSent.method"?: string;
    /** The URL of the webhook to call in response to the `on_message_updated` event using the `webhooks.on_message_updated.method` HTTP method. */
    "webhooks.onMessageUpdated.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_message_updated.url`. */
    "webhooks.onMessageUpdated.method"?: string;
    /** The URL of the webhook to call in response to the `on_message_removed` event using the `webhooks.on_message_removed.method` HTTP method. */
    "webhooks.onMessageRemoved.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_message_removed.url`. */
    "webhooks.onMessageRemoved.method"?: string;
    /** The URL of the webhook to call in response to the `on_channel_added` event using the `webhooks.on_channel_added.method` HTTP method. */
    "webhooks.onChannelAdded.url"?: string;
    /** The URL of the webhook to call in response to the `on_channel_added` event`. */
    "webhooks.onChannelAdded.method"?: string;
    /** The URL of the webhook to call in response to the `on_channel_added` event using the `webhooks.on_channel_destroyed.method` HTTP method. */
    "webhooks.onChannelDestroyed.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_channel_destroyed.url`. */
    "webhooks.onChannelDestroyed.method"?: string;
    /** The URL of the webhook to call in response to the `on_channel_updated` event using the `webhooks.on_channel_updated.method` HTTP method. */
    "webhooks.onChannelUpdated.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_channel_updated.url`. */
    "webhooks.onChannelUpdated.method"?: string;
    /** The URL of the webhook to call in response to the `on_channel_updated` event using the `webhooks.on_channel_updated.method` HTTP method. */
    "webhooks.onMemberAdded.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_channel_updated.url`. */
    "webhooks.onMemberAdded.method"?: string;
    /** The URL of the webhook to call in response to the `on_member_removed` event using the `webhooks.on_member_removed.method` HTTP method. */
    "webhooks.onMemberRemoved.url"?: string;
    /** The HTTP method to use when calling the `webhooks.on_member_removed.url`. */
    "webhooks.onMemberRemoved.method"?: string;
    /** The maximum number of Members that can be added to Channels within this Service. Can be up to 1,000. */
    "limits.channelMembers"?: number;
    /** The maximum number of Channels Users can be a Member of within this Service. Can be up to 1,000. */
    "limits.userChannels"?: number;
}
/**
 * Options to pass to create a ServiceInstance
 */
export interface ServiceListInstanceCreateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
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
    /**
     * The unique string that we created to identify the Service resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/api/rest/account) that created the Service resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](http://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](http://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The service role assigned to users when they are added to the service. See the [Roles endpoint](https://www.twilio.com/docs/chat/api/roles) for more details.
     */
    defaultServiceRoleSid: string;
    /**
     * The channel role assigned to users when they are added to a channel. See the [Roles endpoint](https://www.twilio.com/docs/chat/api/roles) for more details.
     */
    defaultChannelRoleSid: string;
    /**
     * The channel role assigned to a channel creator when they join a new channel. See the [Roles endpoint](https://www.twilio.com/docs/chat/api/roles) for more details.
     */
    defaultChannelCreatorRoleSid: string;
    /**
     * Whether the [Message Consumption Horizon](https://www.twilio.com/docs/chat/consumption-horizon) feature is enabled. The default is `true`.
     */
    readStatusEnabled: boolean;
    /**
     * Whether the [Reachability Indicator](https://www.twilio.com/docs/chat/reachability-indicator) is enabled for this Service instance. The default is `false`.
     */
    reachabilityEnabled: boolean;
    /**
     * How long in seconds after a `started typing` event until clients should assume that user is no longer typing, even if no `ended typing` message was received.  The default is 5 seconds.
     */
    typingIndicatorTimeout: number;
    /**
     * DEPRECATED. The interval in seconds between consumption reports submission batches from client endpoints.
     */
    consumptionReportInterval: number;
    /**
     * An object that describes the limits of the service instance. The `limits` object contains  `channel_members` to describe the members/channel limit and `user_channels` to describe the channels/user limit. `channel_members` can be 1,000 or less, with a default of 250. `user_channels` can be 1,000 or less, with a default value of 100.
     */
    limits: any;
    /**
     * An object that contains information about the webhooks configured for this service.
     */
    webhooks: any;
    /**
     * The URL for pre-event webhooks, which are called by using the `webhook_method`. See [Webhook Events](https://www.twilio.com/docs/api/chat/webhooks) for more details.
     */
    preWebhookUrl: string;
    /**
     * The URL for post-event webhooks, which are called by using the `webhook_method`. See [Webhook Events](https://www.twilio.com/docs/api/chat/webhooks) for more details.
     */
    postWebhookUrl: string;
    /**
     * The HTTP method to use for calls to the `pre_webhook_url` and `post_webhook_url` webhooks.  Can be: `POST` or `GET` and the default is `POST`. See [Webhook Events](https://www.twilio.com/docs/chat/webhook-events) for more details.
     */
    webhookMethod: string;
    /**
     * The list of WebHook events that are enabled for this Service instance. See [Webhook Events](https://www.twilio.com/docs/chat/webhook-events) for more details.
     */
    webhookFilters: Array<string>;
    /**
     * The notification configuration for the Service instance. See [Push Notification Configuration](https://www.twilio.com/docs/chat/push-notification-configuration) for more information.
     */
    notifications: any;
    /**
     * The absolute URL of the Service resource.
     */
    url: string;
    /**
     * The absolute URLs of the Service\'s [Channels](https://www.twilio.com/docs/chat/api/channels), [Roles](https://www.twilio.com/docs/chat/api/roles), and [Users](https://www.twilio.com/docs/chat/api/users).
     */
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
