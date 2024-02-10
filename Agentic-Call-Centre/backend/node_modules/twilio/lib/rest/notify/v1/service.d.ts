/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { BindingListInstance } from "./service/binding";
import { NotificationListInstance } from "./service/notification";
/**
 * Options to pass to update a ServiceInstance
 */
export interface ServiceContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** The SID of the [Credential](https://www.twilio.com/docs/notify/api/credential-resource) to use for APN Bindings. */
    apnCredentialSid?: string;
    /** The SID of the [Credential](https://www.twilio.com/docs/notify/api/credential-resource) to use for GCM Bindings. */
    gcmCredentialSid?: string;
    /** The SID of the [Messaging Service](https://www.twilio.com/docs/sms/quickstart#messaging-services) to use for SMS Bindings. This parameter must be set in order to send SMS notifications. */
    messagingServiceSid?: string;
    /** Deprecated. */
    facebookMessengerPageId?: string;
    /** The protocol version to use for sending APNS notifications. Can be overridden on a Binding by Binding basis when creating a [Binding](https://www.twilio.com/docs/notify/api/binding-resource) resource. */
    defaultApnNotificationProtocolVersion?: string;
    /** The protocol version to use for sending GCM notifications. Can be overridden on a Binding by Binding basis when creating a [Binding](https://www.twilio.com/docs/notify/api/binding-resource) resource. */
    defaultGcmNotificationProtocolVersion?: string;
    /** The SID of the [Credential](https://www.twilio.com/docs/notify/api/credential-resource) to use for FCM Bindings. */
    fcmCredentialSid?: string;
    /** The protocol version to use for sending FCM notifications. Can be overridden on a Binding by Binding basis when creating a [Binding](https://www.twilio.com/docs/notify/api/binding-resource) resource. */
    defaultFcmNotificationProtocolVersion?: string;
    /** Whether to log notifications. Can be: `true` or `false` and the default is `true`. */
    logEnabled?: boolean;
    /** Deprecated. */
    alexaSkillId?: string;
    /** Deprecated. */
    defaultAlexaNotificationProtocolVersion?: string;
    /** URL to send delivery status callback. */
    deliveryCallbackUrl?: string;
    /** Callback configuration that enables delivery callbacks, default false */
    deliveryCallbackEnabled?: boolean;
}
/**
 * Options to pass to create a ServiceInstance
 */
export interface ServiceListInstanceCreateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** The SID of the [Credential](https://www.twilio.com/docs/notify/api/credential-resource) to use for APN Bindings. */
    apnCredentialSid?: string;
    /** The SID of the [Credential](https://www.twilio.com/docs/notify/api/credential-resource) to use for GCM Bindings. */
    gcmCredentialSid?: string;
    /** The SID of the [Messaging Service](https://www.twilio.com/docs/sms/quickstart#messaging-services) to use for SMS Bindings. This parameter must be set in order to send SMS notifications. */
    messagingServiceSid?: string;
    /** Deprecated. */
    facebookMessengerPageId?: string;
    /** The protocol version to use for sending APNS notifications. Can be overridden on a Binding by Binding basis when creating a [Binding](https://www.twilio.com/docs/notify/api/binding-resource) resource. */
    defaultApnNotificationProtocolVersion?: string;
    /** The protocol version to use for sending GCM notifications. Can be overridden on a Binding by Binding basis when creating a [Binding](https://www.twilio.com/docs/notify/api/binding-resource) resource. */
    defaultGcmNotificationProtocolVersion?: string;
    /** The SID of the [Credential](https://www.twilio.com/docs/notify/api/credential-resource) to use for FCM Bindings. */
    fcmCredentialSid?: string;
    /** The protocol version to use for sending FCM notifications. Can be overridden on a Binding by Binding basis when creating a [Binding](https://www.twilio.com/docs/notify/api/binding-resource) resource. */
    defaultFcmNotificationProtocolVersion?: string;
    /** Whether to log notifications. Can be: `true` or `false` and the default is `true`. */
    logEnabled?: boolean;
    /** Deprecated. */
    alexaSkillId?: string;
    /** Deprecated. */
    defaultAlexaNotificationProtocolVersion?: string;
    /** URL to send delivery status callback. */
    deliveryCallbackUrl?: string;
    /** Callback configuration that enables delivery callbacks, default false */
    deliveryCallbackEnabled?: boolean;
}
/**
 * Options to pass to each
 */
export interface ServiceListInstanceEachOptions {
    /** The string that identifies the Service resources to read. */
    friendlyName?: string;
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
    /** The string that identifies the Service resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ServiceListInstancePageOptions {
    /** The string that identifies the Service resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ServiceContext {
    bindings: BindingListInstance;
    notifications: NotificationListInstance;
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
    protected _bindings?: BindingListInstance;
    protected _notifications?: NotificationListInstance;
    constructor(_version: V1, sid: string);
    get bindings(): BindingListInstance;
    get notifications(): NotificationListInstance;
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
    apn_credential_sid: string;
    gcm_credential_sid: string;
    fcm_credential_sid: string;
    messaging_service_sid: string;
    facebook_messenger_page_id: string;
    default_apn_notification_protocol_version: string;
    default_gcm_notification_protocol_version: string;
    default_fcm_notification_protocol_version: string;
    log_enabled: boolean;
    url: string;
    links: Record<string, string>;
    alexa_skill_id: string;
    default_alexa_notification_protocol_version: string;
    delivery_callback_url: string;
    delivery_callback_enabled: boolean;
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
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Service resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The SID of the [Credential](https://www.twilio.com/docs/notify/api/credential-resource) to use for APN Bindings.
     */
    apnCredentialSid: string;
    /**
     * The SID of the [Credential](https://www.twilio.com/docs/notify/api/credential-resource) to use for GCM Bindings.
     */
    gcmCredentialSid: string;
    /**
     * The SID of the [Credential](https://www.twilio.com/docs/notify/api/credential-resource) to use for FCM Bindings.
     */
    fcmCredentialSid: string;
    /**
     * The SID of the [Messaging Service](https://www.twilio.com/docs/sms/quickstart#messaging-services) to use for SMS Bindings. In order to send SMS notifications this parameter has to be set.
     */
    messagingServiceSid: string;
    /**
     * Deprecated.
     */
    facebookMessengerPageId: string;
    /**
     * The protocol version to use for sending APNS notifications. Can be overridden on a Binding by Binding basis when creating a [Binding](https://www.twilio.com/docs/notify/api/binding-resource) resource.
     */
    defaultApnNotificationProtocolVersion: string;
    /**
     * The protocol version to use for sending GCM notifications. Can be overridden on a Binding by Binding basis when creating a [Binding](https://www.twilio.com/docs/notify/api/binding-resource) resource.
     */
    defaultGcmNotificationProtocolVersion: string;
    /**
     * The protocol version to use for sending FCM notifications. Can be overridden on a Binding by Binding basis when creating a [Binding](https://www.twilio.com/docs/notify/api/binding-resource) resource.
     */
    defaultFcmNotificationProtocolVersion: string;
    /**
     * Whether to log notifications. Can be: `true` or `false` and the default is `true`.
     */
    logEnabled: boolean;
    /**
     * The absolute URL of the Service resource.
     */
    url: string;
    /**
     * The URLs of the Binding, Notification, Segment, and User resources related to the service.
     */
    links: Record<string, string>;
    /**
     * Deprecated.
     */
    alexaSkillId: string;
    /**
     * Deprecated.
     */
    defaultAlexaNotificationProtocolVersion: string;
    /**
     * URL to send delivery status callback.
     */
    deliveryCallbackUrl: string;
    /**
     * Callback configuration that enables delivery callbacks, default false
     */
    deliveryCallbackEnabled: boolean;
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
     * Access the bindings.
     */
    bindings(): BindingListInstance;
    /**
     * Access the notifications.
     */
    notifications(): NotificationListInstance;
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
        apnCredentialSid: string;
        gcmCredentialSid: string;
        fcmCredentialSid: string;
        messagingServiceSid: string;
        facebookMessengerPageId: string;
        defaultApnNotificationProtocolVersion: string;
        defaultGcmNotificationProtocolVersion: string;
        defaultFcmNotificationProtocolVersion: string;
        logEnabled: boolean;
        url: string;
        links: Record<string, string>;
        alexaSkillId: string;
        defaultAlexaNotificationProtocolVersion: string;
        deliveryCallbackUrl: string;
        deliveryCallbackEnabled: boolean;
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
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    create(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
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
