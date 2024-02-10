/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { PhoneNumberListInstance } from "./service/phoneNumber";
import { SessionListInstance } from "./service/session";
import { ShortCodeListInstance } from "./service/shortCode";
export type ServiceGeoMatchLevel = "area-code" | "overlay" | "radius" | "country";
export type ServiceNumberSelectionBehavior = "avoid-sticky" | "prefer-sticky";
/**
 * Options to pass to update a ServiceInstance
 */
export interface ServiceContextUpdateOptions {
    /** An application-defined string that uniquely identifies the resource. This value must be 191 characters or fewer in length and be unique. **This value should not have PII.** */
    uniqueName?: string;
    /** The default `ttl` value to set for Sessions created in the Service. The TTL (time to live) is measured in seconds after the Session\\\'s last create or last Interaction. The default value of `0` indicates an unlimited Session length. You can override a Session\\\'s default TTL value by setting its `ttl` value. */
    defaultTtl?: number;
    /** The URL we should call when the interaction status changes. */
    callbackUrl?: string;
    /**  */
    geoMatchLevel?: ServiceGeoMatchLevel;
    /**  */
    numberSelectionBehavior?: ServiceNumberSelectionBehavior;
    /** The URL we call on each interaction. If we receive a 403 status, we block the interaction; otherwise the interaction continues. */
    interceptCallbackUrl?: string;
    /** The URL we should call when an inbound call or SMS action occurs on a closed or non-existent Session. If your server (or a Twilio [function](https://www.twilio.com/en-us/serverless/functions)) responds with valid [TwiML](https://www.twilio.com/docs/voice/twiml), we will process it. This means it is possible, for example, to play a message for a call, send an automated text message response, or redirect a call to another Phone Number. See [Out-of-Session Callback Response Guide](https://www.twilio.com/docs/proxy/out-session-callback-response-guide) for more information. */
    outOfSessionCallbackUrl?: string;
    /** The SID of the Chat Service Instance managed by Proxy Service. The Chat Service enables Proxy to forward SMS and channel messages to this chat instance. This is a one-to-one relationship. */
    chatInstanceSid?: string;
}
/**
 * Options to pass to create a ServiceInstance
 */
export interface ServiceListInstanceCreateOptions {
    /** An application-defined string that uniquely identifies the resource. This value must be 191 characters or fewer in length and be unique. **This value should not have PII.** */
    uniqueName: string;
    /** The default `ttl` value to set for Sessions created in the Service. The TTL (time to live) is measured in seconds after the Session\\\'s last create or last Interaction. The default value of `0` indicates an unlimited Session length. You can override a Session\\\'s default TTL value by setting its `ttl` value. */
    defaultTtl?: number;
    /** The URL we should call when the interaction status changes. */
    callbackUrl?: string;
    /**  */
    geoMatchLevel?: ServiceGeoMatchLevel;
    /**  */
    numberSelectionBehavior?: ServiceNumberSelectionBehavior;
    /** The URL we call on each interaction. If we receive a 403 status, we block the interaction; otherwise the interaction continues. */
    interceptCallbackUrl?: string;
    /** The URL we should call when an inbound call or SMS action occurs on a closed or non-existent Session. If your server (or a Twilio [function](https://www.twilio.com/en-us/serverless/functions)) responds with valid [TwiML](https://www.twilio.com/docs/voice/twiml), we will process it. This means it is possible, for example, to play a message for a call, send an automated text message response, or redirect a call to another Phone Number. See [Out-of-Session Callback Response Guide](https://www.twilio.com/docs/proxy/out-session-callback-response-guide) for more information. */
    outOfSessionCallbackUrl?: string;
    /** The SID of the Chat Service Instance managed by Proxy Service. The Chat Service enables Proxy to forward SMS and channel messages to this chat instance. This is a one-to-one relationship. */
    chatInstanceSid?: string;
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
    phoneNumbers: PhoneNumberListInstance;
    sessions: SessionListInstance;
    shortCodes: ShortCodeListInstance;
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
    protected _phoneNumbers?: PhoneNumberListInstance;
    protected _sessions?: SessionListInstance;
    protected _shortCodes?: ShortCodeListInstance;
    constructor(_version: V1, sid: string);
    get phoneNumbers(): PhoneNumberListInstance;
    get sessions(): SessionListInstance;
    get shortCodes(): ShortCodeListInstance;
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
    unique_name: string;
    account_sid: string;
    chat_instance_sid: string;
    callback_url: string;
    default_ttl: number;
    number_selection_behavior: ServiceNumberSelectionBehavior;
    geo_match_level: ServiceGeoMatchLevel;
    intercept_callback_url: string;
    out_of_session_callback_url: string;
    date_created: Date;
    date_updated: Date;
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
     * An application-defined string that uniquely identifies the resource. This value must be 191 characters or fewer in length and be unique. Supports UTF-8 characters. **This value should not have PII.**
     */
    uniqueName: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Service resource.
     */
    accountSid: string;
    /**
     * The SID of the Chat Service Instance managed by Proxy Service. The Chat Service enables Proxy to forward SMS and channel messages to this chat instance. This is a one-to-one relationship.
     */
    chatInstanceSid: string;
    /**
     * The URL we call when the interaction status changes.
     */
    callbackUrl: string;
    /**
     * The default `ttl` value for Sessions created in the Service. The TTL (time to live) is measured in seconds after the Session\'s last create or last Interaction. The default value of `0` indicates an unlimited Session length. You can override a Session\'s default TTL value by setting its `ttl` value.
     */
    defaultTtl: number;
    numberSelectionBehavior: ServiceNumberSelectionBehavior;
    geoMatchLevel: ServiceGeoMatchLevel;
    /**
     * The URL we call on each interaction. If we receive a 403 status, we block the interaction; otherwise the interaction continues.
     */
    interceptCallbackUrl: string;
    /**
     * The URL we call when an inbound call or SMS action occurs on a closed or non-existent Session. If your server (or a Twilio [function](https://www.twilio.com/en-us/serverless/functions)) responds with valid [TwiML](https://www.twilio.com/docs/voice/twiml), we will process it. This means it is possible, for example, to play a message for a call, send an automated text message response, or redirect a call to another Phone Number. See [Out-of-Session Callback Response Guide](https://www.twilio.com/docs/proxy/out-session-callback-response-guide) for more information.
     */
    outOfSessionCallbackUrl: string;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the resource was created.
     */
    dateCreated: Date;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the resource was last updated.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Service resource.
     */
    url: string;
    /**
     * The URLs of resources related to the Service.
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
     * Access the phoneNumbers.
     */
    phoneNumbers(): PhoneNumberListInstance;
    /**
     * Access the sessions.
     */
    sessions(): SessionListInstance;
    /**
     * Access the shortCodes.
     */
    shortCodes(): ShortCodeListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        uniqueName: string;
        accountSid: string;
        chatInstanceSid: string;
        callbackUrl: string;
        defaultTtl: number;
        numberSelectionBehavior: ServiceNumberSelectionBehavior;
        geoMatchLevel: ServiceGeoMatchLevel;
        interceptCallbackUrl: string;
        outOfSessionCallbackUrl: string;
        dateCreated: Date;
        dateUpdated: Date;
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
