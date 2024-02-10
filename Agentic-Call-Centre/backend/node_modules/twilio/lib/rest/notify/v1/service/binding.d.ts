/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type BindingBindingType = "apn" | "gcm" | "sms" | "fcm" | "facebook-messenger" | "alexa";
/**
 * Options to pass to create a BindingInstance
 */
export interface BindingListInstanceCreateOptions {
    /** The `identity` value that uniquely identifies the new resource\\\'s [User](https://www.twilio.com/docs/chat/rest/user-resource) within the [Service](https://www.twilio.com/docs/notify/api/service-resource). Up to 20 Bindings can be created for the same Identity in a given Service. */
    identity: string;
    /**  */
    bindingType: BindingBindingType;
    /** The channel-specific address. For APNS, the device token. For FCM and GCM, the registration token. For SMS, a phone number in E.164 format. For Facebook Messenger, the Messenger ID of the user or a phone number in E.164 format. */
    address: string;
    /** A tag that can be used to select the Bindings to notify. Repeat this parameter to specify more than one tag, up to a total of 20 tags. */
    tag?: Array<string>;
    /** The protocol version to use to send the notification. This defaults to the value of `default_xxxx_notification_protocol_version` for the protocol in the [Service](https://www.twilio.com/docs/notify/api/service-resource). The current version is `\\\"3\\\"` for `apn`, `fcm`, and `gcm` type Bindings. The parameter is not applicable to `sms` and `facebook-messenger` type Bindings as the data format is fixed. */
    notificationProtocolVersion?: string;
    /** The SID of the [Credential](https://www.twilio.com/docs/notify/api/credential-resource) resource to be used to send notifications to this Binding. If present, this overrides the Credential specified in the Service resource. Applies to only `apn`, `fcm`, and `gcm` type Bindings. */
    credentialSid?: string;
    /** Deprecated. */
    endpoint?: string;
}
/**
 * Options to pass to each
 */
export interface BindingListInstanceEachOptions {
    /** Only include usage that has occurred on or after this date. Specify the date in GMT and format as `YYYY-MM-DD`. */
    startDate?: Date;
    /** Only include usage that occurred on or before this date. Specify the date in GMT and format as `YYYY-MM-DD`. */
    endDate?: Date;
    /** The [User](https://www.twilio.com/docs/chat/rest/user-resource)\'s `identity` value of the resources to read. */
    identity?: Array<string>;
    /** Only list Bindings that have all of the specified Tags. The following implicit tags are available: `all`, `apn`, `fcm`, `gcm`, `sms`, `facebook-messenger`. Up to 5 tags are allowed. */
    tag?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: BindingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface BindingListInstanceOptions {
    /** Only include usage that has occurred on or after this date. Specify the date in GMT and format as `YYYY-MM-DD`. */
    startDate?: Date;
    /** Only include usage that occurred on or before this date. Specify the date in GMT and format as `YYYY-MM-DD`. */
    endDate?: Date;
    /** The [User](https://www.twilio.com/docs/chat/rest/user-resource)\'s `identity` value of the resources to read. */
    identity?: Array<string>;
    /** Only list Bindings that have all of the specified Tags. The following implicit tags are available: `all`, `apn`, `fcm`, `gcm`, `sms`, `facebook-messenger`. Up to 5 tags are allowed. */
    tag?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface BindingListInstancePageOptions {
    /** Only include usage that has occurred on or after this date. Specify the date in GMT and format as `YYYY-MM-DD`. */
    startDate?: Date;
    /** Only include usage that occurred on or before this date. Specify the date in GMT and format as `YYYY-MM-DD`. */
    endDate?: Date;
    /** The [User](https://www.twilio.com/docs/chat/rest/user-resource)\'s `identity` value of the resources to read. */
    identity?: Array<string>;
    /** Only list Bindings that have all of the specified Tags. The following implicit tags are available: `all`, `apn`, `fcm`, `gcm`, `sms`, `facebook-messenger`. Up to 5 tags are allowed. */
    tag?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface BindingContext {
    /**
     * Remove a BindingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a BindingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BindingInstance
     */
    fetch(callback?: (error: Error | null, item?: BindingInstance) => any): Promise<BindingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface BindingContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class BindingContextImpl implements BindingContext {
    protected _version: V1;
    protected _solution: BindingContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: BindingInstance) => any): Promise<BindingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): BindingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface BindingPayload extends TwilioResponsePayload {
    bindings: BindingResource[];
}
interface BindingResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    credential_sid: string;
    date_created: Date;
    date_updated: Date;
    notification_protocol_version: string;
    endpoint: string;
    identity: string;
    binding_type: string;
    address: string;
    tags: Array<string>;
    url: string;
    links: Record<string, string>;
}
export declare class BindingInstance {
    protected _version: V1;
    protected _solution: BindingContextSolution;
    protected _context?: BindingContext;
    constructor(_version: V1, payload: BindingResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Binding resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Binding resource.
     */
    accountSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/notify/api/service-resource) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the [Credential](https://www.twilio.com/docs/notify/api/credential-resource) resource to be used to send notifications to this Binding. If present, this overrides the Credential specified in the Service resource. Applicable only to `apn`, `fcm`, and `gcm` type Bindings.
     */
    credentialSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The protocol version to use to send the notification. This defaults to the value of `default_xxxx_notification_protocol_version` in the [Service](https://www.twilio.com/docs/notify/api/service-resource) for the protocol. The current version is `\"3\"` for `apn`, `fcm`, and `gcm` type Bindings. The parameter is not applicable to `sms` and `facebook-messenger` type Bindings as the data format is fixed.
     */
    notificationProtocolVersion: string;
    /**
     * Deprecated.
     */
    endpoint: string;
    /**
     * The `identity` value that uniquely identifies the resource\'s [User](https://www.twilio.com/docs/chat/rest/user-resource) within the [Service](https://www.twilio.com/docs/notify/api/service-resource). Up to 20 Bindings can be created for the same Identity in a given Service.
     */
    identity: string;
    /**
     * The transport technology to use for the Binding. Can be: `apn`, `fcm`, `gcm`, `sms`, or `facebook-messenger`.
     */
    bindingType: string;
    /**
     * The channel-specific address. For APNS, the device token. For FCM and GCM, the registration token. For SMS, a phone number in E.164 format. For Facebook Messenger, the Messenger ID of the user or a phone number in E.164 format.
     */
    address: string;
    /**
     * The list of tags associated with this Binding. Tags can be used to select the Bindings to use when sending a notification. Maximum 20 tags are allowed.
     */
    tags: Array<string>;
    /**
     * The absolute URL of the Binding resource.
     */
    url: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a BindingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a BindingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BindingInstance
     */
    fetch(callback?: (error: Error | null, item?: BindingInstance) => any): Promise<BindingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        credentialSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        notificationProtocolVersion: string;
        endpoint: string;
        identity: string;
        bindingType: string;
        address: string;
        tags: string[];
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface BindingSolution {
    serviceSid: string;
}
export interface BindingListInstance {
    _version: V1;
    _solution: BindingSolution;
    _uri: string;
    (sid: string): BindingContext;
    get(sid: string): BindingContext;
    /**
     * Create a BindingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BindingInstance
     */
    create(params: BindingListInstanceCreateOptions, callback?: (error: Error | null, item?: BindingInstance) => any): Promise<BindingInstance>;
    /**
     * Streams BindingInstance records from the API.
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
     * @param { BindingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: BindingInstance, done: (err?: Error) => void) => void): void;
    each(params: BindingListInstanceEachOptions, callback?: (item: BindingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of BindingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: BindingPage) => any): Promise<BindingPage>;
    /**
     * Lists BindingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BindingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: BindingInstance[]) => any): Promise<BindingInstance[]>;
    list(params: BindingListInstanceOptions, callback?: (error: Error | null, items: BindingInstance[]) => any): Promise<BindingInstance[]>;
    /**
     * Retrieve a single page of BindingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BindingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: BindingPage) => any): Promise<BindingPage>;
    page(params: BindingListInstancePageOptions, callback?: (error: Error | null, items: BindingPage) => any): Promise<BindingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BindingListInstance(version: V1, serviceSid: string): BindingListInstance;
export declare class BindingPage extends Page<V1, BindingPayload, BindingResource, BindingInstance> {
    /**
     * Initialize the BindingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: BindingSolution);
    /**
     * Build an instance of BindingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: BindingResource): BindingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
