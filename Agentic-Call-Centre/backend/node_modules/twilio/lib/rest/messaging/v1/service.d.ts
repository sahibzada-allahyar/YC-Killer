/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { AlphaSenderListInstance } from "./service/alphaSender";
import { ChannelSenderListInstance } from "./service/channelSender";
import { PhoneNumberListInstance } from "./service/phoneNumber";
import { ShortCodeListInstance } from "./service/shortCode";
import { UsAppToPersonListInstance } from "./service/usAppToPerson";
import { UsAppToPersonUsecaseListInstance } from "./service/usAppToPersonUsecase";
export type ServiceScanMessageContent = "inherit" | "enable" | "disable";
/**
 * Options to pass to update a ServiceInstance
 */
export interface ServiceContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** The URL we call using `inbound_method` when a message is received by any phone number or short code in the Service. When this property is `null`, receiving inbound messages is disabled. All messages sent to the Twilio phone number or short code will not be logged and received on the Account. If the `use_inbound_webhook_on_number` field is enabled then the webhook url defined on the phone number will override the `inbound_request_url` defined for the Messaging Service. */
    inboundRequestUrl?: string;
    /** The HTTP method we should use to call `inbound_request_url`. Can be `GET` or `POST` and the default is `POST`. */
    inboundMethod?: string;
    /** The URL that we call using `fallback_method` if an error occurs while retrieving or executing the TwiML from the Inbound Request URL. If the `use_inbound_webhook_on_number` field is enabled then the webhook url defined on the phone number will override the `fallback_url` defined for the Messaging Service. */
    fallbackUrl?: string;
    /** The HTTP method we should use to call `fallback_url`. Can be: `GET` or `POST`. */
    fallbackMethod?: string;
    /** The URL we should call to [pass status updates](https://www.twilio.com/docs/sms/api/message-resource#message-status-values) about message delivery. */
    statusCallback?: string;
    /** Whether to enable [Sticky Sender](https://www.twilio.com/docs/messaging/services#sticky-sender) on the Service instance. */
    stickySender?: boolean;
    /** Whether to enable the [MMS Converter](https://www.twilio.com/docs/messaging/services#mms-converter) for messages sent through the Service instance. */
    mmsConverter?: boolean;
    /** Whether to enable [Smart Encoding](https://www.twilio.com/docs/messaging/services#smart-encoding) for messages sent through the Service instance. */
    smartEncoding?: boolean;
    /**  */
    scanMessageContent?: ServiceScanMessageContent;
    /** [OBSOLETE] Former feature used to fallback to long code sender after certain short code message failures. */
    fallbackToLongCode?: boolean;
    /** Whether to enable [Area Code Geomatch](https://www.twilio.com/docs/messaging/services#area-code-geomatch) on the Service Instance. */
    areaCodeGeomatch?: boolean;
    /** How long, in seconds, messages sent from the Service are valid. Can be an integer from `1` to `14,400`. */
    validityPeriod?: number;
    /** Reserved. */
    synchronousValidation?: boolean;
    /** A string that describes the scenario in which the Messaging Service will be used. Possible values are `notifications`, `marketing`, `verification`, `discussion`, `poll`, `undeclared`. */
    usecase?: string;
    /** A boolean value that indicates either the webhook url configured on the phone number will be used or `inbound_request_url`/`fallback_url` url will be called when a message is received from the phone number. If this field is enabled then the webhook url defined on the phone number will override the `inbound_request_url`/`fallback_url` defined for the Messaging Service. */
    useInboundWebhookOnNumber?: boolean;
}
/**
 * Options to pass to create a ServiceInstance
 */
export interface ServiceListInstanceCreateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName: string;
    /** The URL we call using `inbound_method` when a message is received by any phone number or short code in the Service. When this property is `null`, receiving inbound messages is disabled. All messages sent to the Twilio phone number or short code will not be logged and received on the Account. If the `use_inbound_webhook_on_number` field is enabled then the webhook url defined on the phone number will override the `inbound_request_url` defined for the Messaging Service. */
    inboundRequestUrl?: string;
    /** The HTTP method we should use to call `inbound_request_url`. Can be `GET` or `POST` and the default is `POST`. */
    inboundMethod?: string;
    /** The URL that we call using `fallback_method` if an error occurs while retrieving or executing the TwiML from the Inbound Request URL. If the `use_inbound_webhook_on_number` field is enabled then the webhook url defined on the phone number will override the `fallback_url` defined for the Messaging Service. */
    fallbackUrl?: string;
    /** The HTTP method we should use to call `fallback_url`. Can be: `GET` or `POST`. */
    fallbackMethod?: string;
    /** The URL we should call to [pass status updates](https://www.twilio.com/docs/sms/api/message-resource#message-status-values) about message delivery. */
    statusCallback?: string;
    /** Whether to enable [Sticky Sender](https://www.twilio.com/docs/messaging/services#sticky-sender) on the Service instance. */
    stickySender?: boolean;
    /** Whether to enable the [MMS Converter](https://www.twilio.com/docs/messaging/services#mms-converter) for messages sent through the Service instance. */
    mmsConverter?: boolean;
    /** Whether to enable [Smart Encoding](https://www.twilio.com/docs/messaging/services#smart-encoding) for messages sent through the Service instance. */
    smartEncoding?: boolean;
    /**  */
    scanMessageContent?: ServiceScanMessageContent;
    /** [OBSOLETE] Former feature used to fallback to long code sender after certain short code message failures. */
    fallbackToLongCode?: boolean;
    /** Whether to enable [Area Code Geomatch](https://www.twilio.com/docs/messaging/services#area-code-geomatch) on the Service Instance. */
    areaCodeGeomatch?: boolean;
    /** How long, in seconds, messages sent from the Service are valid. Can be an integer from `1` to `14,400`. */
    validityPeriod?: number;
    /** Reserved. */
    synchronousValidation?: boolean;
    /** A string that describes the scenario in which the Messaging Service will be used. Possible values are `notifications`, `marketing`, `verification`, `discussion`, `poll`, `undeclared`. */
    usecase?: string;
    /** A boolean value that indicates either the webhook url configured on the phone number will be used or `inbound_request_url`/`fallback_url` url will be called when a message is received from the phone number. If this field is enabled then the webhook url defined on the phone number will override the `inbound_request_url`/`fallback_url` defined for the Messaging Service. */
    useInboundWebhookOnNumber?: boolean;
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
    alphaSenders: AlphaSenderListInstance;
    channelSenders: ChannelSenderListInstance;
    phoneNumbers: PhoneNumberListInstance;
    shortCodes: ShortCodeListInstance;
    usAppToPerson: UsAppToPersonListInstance;
    usAppToPersonUsecases: UsAppToPersonUsecaseListInstance;
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
    protected _alphaSenders?: AlphaSenderListInstance;
    protected _channelSenders?: ChannelSenderListInstance;
    protected _phoneNumbers?: PhoneNumberListInstance;
    protected _shortCodes?: ShortCodeListInstance;
    protected _usAppToPerson?: UsAppToPersonListInstance;
    protected _usAppToPersonUsecases?: UsAppToPersonUsecaseListInstance;
    constructor(_version: V1, sid: string);
    get alphaSenders(): AlphaSenderListInstance;
    get channelSenders(): ChannelSenderListInstance;
    get phoneNumbers(): PhoneNumberListInstance;
    get shortCodes(): ShortCodeListInstance;
    get usAppToPerson(): UsAppToPersonListInstance;
    get usAppToPersonUsecases(): UsAppToPersonUsecaseListInstance;
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
    inbound_request_url: string;
    inbound_method: string;
    fallback_url: string;
    fallback_method: string;
    status_callback: string;
    sticky_sender: boolean;
    mms_converter: boolean;
    smart_encoding: boolean;
    scan_message_content: ServiceScanMessageContent;
    fallback_to_long_code: boolean;
    area_code_geomatch: boolean;
    synchronous_validation: boolean;
    validity_period: number;
    url: string;
    links: Record<string, string>;
    usecase: string;
    us_app_to_person_registered: boolean;
    use_inbound_webhook_on_number: boolean;
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
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The URL we call using `inbound_method` when a message is received by any phone number or short code in the Service. When this property is `null`, receiving inbound messages is disabled. All messages sent to the Twilio phone number or short code will not be logged and received on the Account. If the `use_inbound_webhook_on_number` field is enabled then the webhook url defined on the phone number will override the `inbound_request_url` defined for the Messaging Service.
     */
    inboundRequestUrl: string;
    /**
     * The HTTP method we use to call `inbound_request_url`. Can be `GET` or `POST`.
     */
    inboundMethod: string;
    /**
     * The URL that we call using `fallback_method` if an error occurs while retrieving or executing the TwiML from the Inbound Request URL. If the `use_inbound_webhook_on_number` field is enabled then the webhook url defined on the phone number will override the `fallback_url` defined for the Messaging Service.
     */
    fallbackUrl: string;
    /**
     * The HTTP method we use to call `fallback_url`. Can be: `GET` or `POST`.
     */
    fallbackMethod: string;
    /**
     * The URL we call to [pass status updates](https://www.twilio.com/docs/sms/api/message-resource#message-status-values) about message delivery.
     */
    statusCallback: string;
    /**
     * Whether to enable [Sticky Sender](https://www.twilio.com/docs/messaging/services#sticky-sender) on the Service instance.
     */
    stickySender: boolean;
    /**
     * Whether to enable the [MMS Converter](https://www.twilio.com/docs/messaging/services#mms-converter) for messages sent through the Service instance.
     */
    mmsConverter: boolean;
    /**
     * Whether to enable [Smart Encoding](https://www.twilio.com/docs/messaging/services#smart-encoding) for messages sent through the Service instance.
     */
    smartEncoding: boolean;
    scanMessageContent: ServiceScanMessageContent;
    /**
     * [OBSOLETE] Former feature used to fallback to long code sender after certain short code message failures.
     */
    fallbackToLongCode: boolean;
    /**
     * Whether to enable [Area Code Geomatch](https://www.twilio.com/docs/messaging/services#area-code-geomatch) on the Service Instance.
     */
    areaCodeGeomatch: boolean;
    /**
     * Reserved.
     */
    synchronousValidation: boolean;
    /**
     * How long, in seconds, messages sent from the Service are valid. Can be an integer from `1` to `14,400`.
     */
    validityPeriod: number;
    /**
     * The absolute URL of the Service resource.
     */
    url: string;
    /**
     * The absolute URLs of related resources.
     */
    links: Record<string, string>;
    /**
     * A string that describes the scenario in which the Messaging Service will be used. Possible values are `notifications`, `marketing`, `verification`, `discussion`, `poll`, `undeclared`.
     */
    usecase: string;
    /**
     * Whether US A2P campaign is registered for this Service.
     */
    usAppToPersonRegistered: boolean;
    /**
     * A boolean value that indicates either the webhook url configured on the phone number will be used or `inbound_request_url`/`fallback_url` url will be called when a message is received from the phone number. If this field is enabled then the webhook url defined on the phone number will override the `inbound_request_url`/`fallback_url` defined for the Messaging Service.
     */
    useInboundWebhookOnNumber: boolean;
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
     * Access the alphaSenders.
     */
    alphaSenders(): AlphaSenderListInstance;
    /**
     * Access the channelSenders.
     */
    channelSenders(): ChannelSenderListInstance;
    /**
     * Access the phoneNumbers.
     */
    phoneNumbers(): PhoneNumberListInstance;
    /**
     * Access the shortCodes.
     */
    shortCodes(): ShortCodeListInstance;
    /**
     * Access the usAppToPerson.
     */
    usAppToPerson(): UsAppToPersonListInstance;
    /**
     * Access the usAppToPersonUsecases.
     */
    usAppToPersonUsecases(): UsAppToPersonUsecaseListInstance;
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
        inboundRequestUrl: string;
        inboundMethod: string;
        fallbackUrl: string;
        fallbackMethod: string;
        statusCallback: string;
        stickySender: boolean;
        mmsConverter: boolean;
        smartEncoding: boolean;
        scanMessageContent: ServiceScanMessageContent;
        fallbackToLongCode: boolean;
        areaCodeGeomatch: boolean;
        synchronousValidation: boolean;
        validityPeriod: number;
        url: string;
        links: Record<string, string>;
        usecase: string;
        usAppToPersonRegistered: boolean;
        useInboundWebhookOnNumber: boolean;
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
