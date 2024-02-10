/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type PhoneNumberAddressRequirement = "none" | "any" | "local" | "foreign";
/**
 * Options to pass to create a PhoneNumberInstance
 */
export interface PhoneNumberListInstanceCreateOptions {
    /** The SID of the [Incoming Phone Number](https://www.twilio.com/docs/phone-numbers/api/incomingphonenumber-resource) that you want to associate with the trunk. */
    phoneNumberSid: string;
}
/**
 * Options to pass to each
 */
export interface PhoneNumberListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: PhoneNumberInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface PhoneNumberListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface PhoneNumberListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface PhoneNumberContext {
    /**
     * Remove a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    fetch(callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PhoneNumberContextSolution {
    trunkSid: string;
    sid: string;
}
export declare class PhoneNumberContextImpl implements PhoneNumberContext {
    protected _version: V1;
    protected _solution: PhoneNumberContextSolution;
    protected _uri: string;
    constructor(_version: V1, trunkSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PhoneNumberContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PhoneNumberPayload extends TwilioResponsePayload {
    phone_numbers: PhoneNumberResource[];
}
interface PhoneNumberResource {
    account_sid: string;
    address_requirements: PhoneNumberAddressRequirement;
    api_version: string;
    beta: boolean;
    capabilities: Record<string, string>;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    links: Record<string, string>;
    phone_number: string;
    sid: string;
    sms_application_sid: string;
    sms_fallback_method: string;
    sms_fallback_url: string;
    sms_method: string;
    sms_url: string;
    status_callback: string;
    status_callback_method: string;
    trunk_sid: string;
    url: string;
    voice_application_sid: string;
    voice_caller_id_lookup: boolean;
    voice_fallback_method: string;
    voice_fallback_url: string;
    voice_method: string;
    voice_url: string;
}
export declare class PhoneNumberInstance {
    protected _version: V1;
    protected _solution: PhoneNumberContextSolution;
    protected _context?: PhoneNumberContext;
    constructor(_version: V1, payload: PhoneNumberResource, trunkSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the PhoneNumber resource.
     */
    accountSid: string;
    addressRequirements: PhoneNumberAddressRequirement;
    /**
     * The API version used to start a new TwiML session.
     */
    apiVersion: string;
    /**
     * Whether the phone number is new to the Twilio platform. Can be: `true` or `false`.
     */
    beta: boolean;
    /**
     * The set of Boolean properties that indicate whether a phone number can receive calls or messages.  Capabilities are  `Voice`, `SMS`, and `MMS` and each capability can be: `true` or `false`.
     */
    capabilities: Record<string, string>;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    /**
     * The phone number in [E.164](https://www.twilio.com/docs/glossary/what-e164) format, which consists of a + followed by the country code and subscriber number.
     */
    phoneNumber: string;
    /**
     * The unique string that we created to identify the PhoneNumber resource.
     */
    sid: string;
    /**
     * The SID of the application that handles SMS messages sent to the phone number. If an `sms_application_sid` is present, we ignore all `sms_*_url` values and use those of the application.
     */
    smsApplicationSid: string;
    /**
     * The HTTP method we use to call `sms_fallback_url`. Can be: `GET` or `POST`.
     */
    smsFallbackMethod: string;
    /**
     * The URL that we call using the `sms_fallback_method` when an error occurs while retrieving or executing the TwiML from `sms_url`.
     */
    smsFallbackUrl: string;
    /**
     * The HTTP method we use to call `sms_url`. Can be: `GET` or `POST`.
     */
    smsMethod: string;
    /**
     * The URL we call using the `sms_method` when the phone number receives an incoming SMS message.
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
     * The SID of the Trunk that handles calls to the phone number. If a `trunk_sid` is present, we ignore all of the voice URLs and voice applications and use those set on the Trunk. Setting a `trunk_sid` will automatically delete your `voice_application_sid` and vice versa.
     */
    trunkSid: string;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    /**
     * The SID of the application that handles calls to the phone number. If a `voice_application_sid` is present, we ignore all of the voice URLs and use those set on the application. Setting a `voice_application_sid` will automatically delete your `trunk_sid` and vice versa.
     */
    voiceApplicationSid: string;
    /**
     * Whether we look up the caller\'s caller-ID name from the CNAM database ($0.01 per look up). Can be: `true` or `false`.
     */
    voiceCallerIdLookup: boolean;
    /**
     * The HTTP method that we use to call `voice_fallback_url`. Can be: `GET` or `POST`.
     */
    voiceFallbackMethod: string;
    /**
     * The URL that we call using the `voice_fallback_method` when an error occurs retrieving or executing the TwiML requested by `url`.
     */
    voiceFallbackUrl: string;
    /**
     * The HTTP method we use to call `voice_url`. Can be: `GET` or `POST`.
     */
    voiceMethod: string;
    /**
     * The URL we call using the `voice_method` when the phone number receives a call. The `voice_url` is not be used if a `voice_application_sid` or a `trunk_sid` is set.
     */
    voiceUrl: string;
    private get _proxy();
    /**
     * Remove a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    fetch(callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        addressRequirements: PhoneNumberAddressRequirement;
        apiVersion: string;
        beta: boolean;
        capabilities: Record<string, string>;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        links: Record<string, string>;
        phoneNumber: string;
        sid: string;
        smsApplicationSid: string;
        smsFallbackMethod: string;
        smsFallbackUrl: string;
        smsMethod: string;
        smsUrl: string;
        statusCallback: string;
        statusCallbackMethod: string;
        trunkSid: string;
        url: string;
        voiceApplicationSid: string;
        voiceCallerIdLookup: boolean;
        voiceFallbackMethod: string;
        voiceFallbackUrl: string;
        voiceMethod: string;
        voiceUrl: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PhoneNumberSolution {
    trunkSid: string;
}
export interface PhoneNumberListInstance {
    _version: V1;
    _solution: PhoneNumberSolution;
    _uri: string;
    (sid: string): PhoneNumberContext;
    get(sid: string): PhoneNumberContext;
    /**
     * Create a PhoneNumberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    create(params: PhoneNumberListInstanceCreateOptions, callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Streams PhoneNumberInstance records from the API.
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
     * @param { PhoneNumberListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: PhoneNumberInstance, done: (err?: Error) => void) => void): void;
    each(params: PhoneNumberListInstanceEachOptions, callback?: (item: PhoneNumberInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of PhoneNumberInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: PhoneNumberPage) => any): Promise<PhoneNumberPage>;
    /**
     * Lists PhoneNumberInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PhoneNumberListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: PhoneNumberInstance[]) => any): Promise<PhoneNumberInstance[]>;
    list(params: PhoneNumberListInstanceOptions, callback?: (error: Error | null, items: PhoneNumberInstance[]) => any): Promise<PhoneNumberInstance[]>;
    /**
     * Retrieve a single page of PhoneNumberInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PhoneNumberListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: PhoneNumberPage) => any): Promise<PhoneNumberPage>;
    page(params: PhoneNumberListInstancePageOptions, callback?: (error: Error | null, items: PhoneNumberPage) => any): Promise<PhoneNumberPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PhoneNumberListInstance(version: V1, trunkSid: string): PhoneNumberListInstance;
export declare class PhoneNumberPage extends Page<V1, PhoneNumberPayload, PhoneNumberResource, PhoneNumberInstance> {
    /**
     * Initialize the PhoneNumberPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: PhoneNumberSolution);
    /**
     * Build an instance of PhoneNumberInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: PhoneNumberResource): PhoneNumberInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
