/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
/**
 * Options to pass to update a ApplicationInstance
 */
export interface ApplicationContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** The API version to use to start a new TwiML session. Can be: `2010-04-01` or `2008-08-01`. The default value is your account\\\'s default API version. */
    apiVersion?: string;
    /** The URL we should call when the phone number assigned to this application receives a call. */
    voiceUrl?: string;
    /** The HTTP method we should use to call `voice_url`. Can be: `GET` or `POST`. */
    voiceMethod?: string;
    /** The URL that we should call when an error occurs retrieving or executing the TwiML requested by `url`. */
    voiceFallbackUrl?: string;
    /** The HTTP method we should use to call `voice_fallback_url`. Can be: `GET` or `POST`. */
    voiceFallbackMethod?: string;
    /** The URL we should call using the `status_callback_method` to send status information to your application. */
    statusCallback?: string;
    /** The HTTP method we should use to call `status_callback`. Can be: `GET` or `POST`. */
    statusCallbackMethod?: string;
    /** Whether we should look up the caller\\\'s caller-ID name from the CNAM database (additional charges apply). Can be: `true` or `false`. */
    voiceCallerIdLookup?: boolean;
    /** The URL we should call when the phone number receives an incoming SMS message. */
    smsUrl?: string;
    /** The HTTP method we should use to call `sms_url`. Can be: `GET` or `POST`. */
    smsMethod?: string;
    /** The URL that we should call when an error occurs while retrieving or executing the TwiML from `sms_url`. */
    smsFallbackUrl?: string;
    /** The HTTP method we should use to call `sms_fallback_url`. Can be: `GET` or `POST`. */
    smsFallbackMethod?: string;
    /** Same as message_status_callback: The URL we should call using a POST method to send status information about SMS messages sent by the application. Deprecated, included for backwards compatibility. */
    smsStatusCallback?: string;
    /** The URL we should call using a POST method to send message status information to your application. */
    messageStatusCallback?: string;
    /** Whether to allow other Twilio accounts to dial this applicaton using Dial verb. Can be: `true` or `false`. */
    publicApplicationConnectEnabled?: boolean;
}
/**
 * Options to pass to create a ApplicationInstance
 */
export interface ApplicationListInstanceCreateOptions {
    /** The API version to use to start a new TwiML session. Can be: `2010-04-01` or `2008-08-01`. The default value is the account\\\'s default API version. */
    apiVersion?: string;
    /** The URL we should call when the phone number assigned to this application receives a call. */
    voiceUrl?: string;
    /** The HTTP method we should use to call `voice_url`. Can be: `GET` or `POST`. */
    voiceMethod?: string;
    /** The URL that we should call when an error occurs retrieving or executing the TwiML requested by `url`. */
    voiceFallbackUrl?: string;
    /** The HTTP method we should use to call `voice_fallback_url`. Can be: `GET` or `POST`. */
    voiceFallbackMethod?: string;
    /** The URL we should call using the `status_callback_method` to send status information to your application. */
    statusCallback?: string;
    /** The HTTP method we should use to call `status_callback`. Can be: `GET` or `POST`. */
    statusCallbackMethod?: string;
    /** Whether we should look up the caller\\\'s caller-ID name from the CNAM database (additional charges apply). Can be: `true` or `false`. */
    voiceCallerIdLookup?: boolean;
    /** The URL we should call when the phone number receives an incoming SMS message. */
    smsUrl?: string;
    /** The HTTP method we should use to call `sms_url`. Can be: `GET` or `POST`. */
    smsMethod?: string;
    /** The URL that we should call when an error occurs while retrieving or executing the TwiML from `sms_url`. */
    smsFallbackUrl?: string;
    /** The HTTP method we should use to call `sms_fallback_url`. Can be: `GET` or `POST`. */
    smsFallbackMethod?: string;
    /** The URL we should call using a POST method to send status information about SMS messages sent by the application. */
    smsStatusCallback?: string;
    /** The URL we should call using a POST method to send message status information to your application. */
    messageStatusCallback?: string;
    /** A descriptive string that you create to describe the new application. It can be up to 64 characters long. */
    friendlyName?: string;
    /** Whether to allow other Twilio accounts to dial this applicaton using Dial verb. Can be: `true` or `false`. */
    publicApplicationConnectEnabled?: boolean;
}
/**
 * Options to pass to each
 */
export interface ApplicationListInstanceEachOptions {
    /** The string that identifies the Application resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ApplicationInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ApplicationListInstanceOptions {
    /** The string that identifies the Application resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ApplicationListInstancePageOptions {
    /** The string that identifies the Application resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ApplicationContext {
    /**
     * Remove a ApplicationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ApplicationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ApplicationInstance
     */
    fetch(callback?: (error: Error | null, item?: ApplicationInstance) => any): Promise<ApplicationInstance>;
    /**
     * Update a ApplicationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ApplicationInstance
     */
    update(callback?: (error: Error | null, item?: ApplicationInstance) => any): Promise<ApplicationInstance>;
    /**
     * Update a ApplicationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ApplicationInstance
     */
    update(params: ApplicationContextUpdateOptions, callback?: (error: Error | null, item?: ApplicationInstance) => any): Promise<ApplicationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ApplicationContextSolution {
    accountSid: string;
    sid: string;
}
export declare class ApplicationContextImpl implements ApplicationContext {
    protected _version: V2010;
    protected _solution: ApplicationContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ApplicationInstance) => any): Promise<ApplicationInstance>;
    update(params?: ApplicationContextUpdateOptions | ((error: Error | null, item?: ApplicationInstance) => any), callback?: (error: Error | null, item?: ApplicationInstance) => any): Promise<ApplicationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ApplicationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ApplicationPayload extends TwilioResponsePayload {
    applications: ApplicationResource[];
}
interface ApplicationResource {
    account_sid: string;
    api_version: string;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    message_status_callback: string;
    sid: string;
    sms_fallback_method: string;
    sms_fallback_url: string;
    sms_method: string;
    sms_status_callback: string;
    sms_url: string;
    status_callback: string;
    status_callback_method: string;
    uri: string;
    voice_caller_id_lookup: boolean;
    voice_fallback_method: string;
    voice_fallback_url: string;
    voice_method: string;
    voice_url: string;
    public_application_connect_enabled: boolean;
}
export declare class ApplicationInstance {
    protected _version: V2010;
    protected _solution: ApplicationContextSolution;
    protected _context?: ApplicationContext;
    constructor(_version: V2010, payload: ApplicationResource, accountSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Application resource.
     */
    accountSid: string;
    /**
     * The API version used to start a new TwiML session.
     */
    apiVersion: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The URL we call using a POST method to send message status information to your application.
     */
    messageStatusCallback: string;
    /**
     * The unique string that that we created to identify the Application resource.
     */
    sid: string;
    /**
     * The HTTP method we use to call `sms_fallback_url`. Can be: `GET` or `POST`.
     */
    smsFallbackMethod: string;
    /**
     * The URL that we call when an error occurs while retrieving or executing the TwiML from `sms_url`.
     */
    smsFallbackUrl: string;
    /**
     * The HTTP method we use to call `sms_url`. Can be: `GET` or `POST`.
     */
    smsMethod: string;
    /**
     * The URL we call using a POST method to send status information to your application about SMS messages that refer to the application.
     */
    smsStatusCallback: string;
    /**
     * The URL we call when the phone number receives an incoming SMS message.
     */
    smsUrl: string;
    /**
     * The URL we call using the `status_callback_method` to send status information to your application.
     */
    statusCallback: string;
    /**
     * The HTTP method we use to call `status_callback`. Can be: `GET` or `POST`.
     */
    statusCallbackMethod: string;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    /**
     * Whether we look up the caller\'s caller-ID name from the CNAM database (additional charges apply). Can be: `true` or `false`.
     */
    voiceCallerIdLookup: boolean;
    /**
     * The HTTP method we use to call `voice_fallback_url`. Can be: `GET` or `POST`.
     */
    voiceFallbackMethod: string;
    /**
     * The URL that we call when an error occurs retrieving or executing the TwiML requested by `url`.
     */
    voiceFallbackUrl: string;
    /**
     * The HTTP method we use to call `voice_url`. Can be: `GET` or `POST`.
     */
    voiceMethod: string;
    /**
     * The URL we call when the phone number assigned to this application receives a call.
     */
    voiceUrl: string;
    /**
     * Whether to allow other Twilio accounts to dial this applicaton using Dial verb. Can be: `true` or `false`.
     */
    publicApplicationConnectEnabled: boolean;
    private get _proxy();
    /**
     * Remove a ApplicationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ApplicationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ApplicationInstance
     */
    fetch(callback?: (error: Error | null, item?: ApplicationInstance) => any): Promise<ApplicationInstance>;
    /**
     * Update a ApplicationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ApplicationInstance
     */
    update(callback?: (error: Error | null, item?: ApplicationInstance) => any): Promise<ApplicationInstance>;
    /**
     * Update a ApplicationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ApplicationInstance
     */
    update(params: ApplicationContextUpdateOptions, callback?: (error: Error | null, item?: ApplicationInstance) => any): Promise<ApplicationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        apiVersion: string;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        messageStatusCallback: string;
        sid: string;
        smsFallbackMethod: string;
        smsFallbackUrl: string;
        smsMethod: string;
        smsStatusCallback: string;
        smsUrl: string;
        statusCallback: string;
        statusCallbackMethod: string;
        uri: string;
        voiceCallerIdLookup: boolean;
        voiceFallbackMethod: string;
        voiceFallbackUrl: string;
        voiceMethod: string;
        voiceUrl: string;
        publicApplicationConnectEnabled: boolean;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ApplicationSolution {
    accountSid: string;
}
export interface ApplicationListInstance {
    _version: V2010;
    _solution: ApplicationSolution;
    _uri: string;
    (sid: string): ApplicationContext;
    get(sid: string): ApplicationContext;
    /**
     * Create a ApplicationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ApplicationInstance
     */
    create(callback?: (error: Error | null, item?: ApplicationInstance) => any): Promise<ApplicationInstance>;
    /**
     * Create a ApplicationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ApplicationInstance
     */
    create(params: ApplicationListInstanceCreateOptions, callback?: (error: Error | null, item?: ApplicationInstance) => any): Promise<ApplicationInstance>;
    /**
     * Streams ApplicationInstance records from the API.
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
     * @param { ApplicationListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ApplicationInstance, done: (err?: Error) => void) => void): void;
    each(params: ApplicationListInstanceEachOptions, callback?: (item: ApplicationInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ApplicationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ApplicationPage) => any): Promise<ApplicationPage>;
    /**
     * Lists ApplicationInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ApplicationListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ApplicationInstance[]) => any): Promise<ApplicationInstance[]>;
    list(params: ApplicationListInstanceOptions, callback?: (error: Error | null, items: ApplicationInstance[]) => any): Promise<ApplicationInstance[]>;
    /**
     * Retrieve a single page of ApplicationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ApplicationListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ApplicationPage) => any): Promise<ApplicationPage>;
    page(params: ApplicationListInstancePageOptions, callback?: (error: Error | null, items: ApplicationPage) => any): Promise<ApplicationPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ApplicationListInstance(version: V2010, accountSid: string): ApplicationListInstance;
export declare class ApplicationPage extends Page<V2010, ApplicationPayload, ApplicationResource, ApplicationInstance> {
    /**
     * Initialize the ApplicationPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: ApplicationSolution);
    /**
     * Build an instance of ApplicationInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ApplicationResource): ApplicationInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
