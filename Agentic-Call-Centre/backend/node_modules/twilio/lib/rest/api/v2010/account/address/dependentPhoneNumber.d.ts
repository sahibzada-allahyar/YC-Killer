/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2010 from "../../../V2010";
export type DependentPhoneNumberAddressRequirement = "none" | "any" | "local" | "foreign";
export type DependentPhoneNumberEmergencyStatus = "Active" | "Inactive";
/**
 * Options to pass to each
 */
export interface DependentPhoneNumberListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: DependentPhoneNumberInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface DependentPhoneNumberListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface DependentPhoneNumberListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface DependentPhoneNumberSolution {
    accountSid: string;
    addressSid: string;
}
export interface DependentPhoneNumberListInstance {
    _version: V2010;
    _solution: DependentPhoneNumberSolution;
    _uri: string;
    /**
     * Streams DependentPhoneNumberInstance records from the API.
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
     * @param { DependentPhoneNumberListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: DependentPhoneNumberInstance, done: (err?: Error) => void) => void): void;
    each(params: DependentPhoneNumberListInstanceEachOptions, callback?: (item: DependentPhoneNumberInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of DependentPhoneNumberInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: DependentPhoneNumberPage) => any): Promise<DependentPhoneNumberPage>;
    /**
     * Lists DependentPhoneNumberInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DependentPhoneNumberListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: DependentPhoneNumberInstance[]) => any): Promise<DependentPhoneNumberInstance[]>;
    list(params: DependentPhoneNumberListInstanceOptions, callback?: (error: Error | null, items: DependentPhoneNumberInstance[]) => any): Promise<DependentPhoneNumberInstance[]>;
    /**
     * Retrieve a single page of DependentPhoneNumberInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DependentPhoneNumberListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: DependentPhoneNumberPage) => any): Promise<DependentPhoneNumberPage>;
    page(params: DependentPhoneNumberListInstancePageOptions, callback?: (error: Error | null, items: DependentPhoneNumberPage) => any): Promise<DependentPhoneNumberPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DependentPhoneNumberListInstance(version: V2010, accountSid: string, addressSid: string): DependentPhoneNumberListInstance;
interface DependentPhoneNumberPayload extends TwilioResponsePayload {
    dependent_phone_numbers: DependentPhoneNumberResource[];
}
interface DependentPhoneNumberResource {
    sid: string;
    account_sid: string;
    friendly_name: string;
    phone_number: string;
    voice_url: string;
    voice_method: string;
    voice_fallback_method: string;
    voice_fallback_url: string;
    voice_caller_id_lookup: boolean;
    date_created: Date;
    date_updated: Date;
    sms_fallback_method: string;
    sms_fallback_url: string;
    sms_method: string;
    sms_url: string;
    address_requirements: DependentPhoneNumberAddressRequirement;
    capabilities: any;
    status_callback: string;
    status_callback_method: string;
    api_version: string;
    sms_application_sid: string;
    voice_application_sid: string;
    trunk_sid: string;
    emergency_status: DependentPhoneNumberEmergencyStatus;
    emergency_address_sid: string;
    uri: string;
}
export declare class DependentPhoneNumberInstance {
    protected _version: V2010;
    constructor(_version: V2010, payload: DependentPhoneNumberResource, accountSid: string, addressSid: string);
    /**
     * The unique string that that we created to identify the DependentPhoneNumber resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the DependentPhoneNumber resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The phone number in [E.164](https://www.twilio.com/docs/glossary/what-e164) format, which consists of a + followed by the country code and subscriber number.
     */
    phoneNumber: string;
    /**
     * The URL we call when the phone number receives a call. The `voice_url` will not be used if a `voice_application_sid` or a `trunk_sid` is set.
     */
    voiceUrl: string;
    /**
     * The HTTP method we use to call `voice_url`. Can be: `GET` or `POST`.
     */
    voiceMethod: string;
    /**
     * The HTTP method we use to call `voice_fallback_url`. Can be: `GET` or `POST`.
     */
    voiceFallbackMethod: string;
    /**
     * The URL that we call when an error occurs retrieving or executing the TwiML requested by `url`.
     */
    voiceFallbackUrl: string;
    /**
     * Whether we look up the caller\'s caller-ID name from the CNAM database. Can be: `true` or `false`. Caller ID lookups can cost $0.01 each.
     */
    voiceCallerIdLookup: boolean;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
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
     * The URL we call when the phone number receives an incoming SMS message.
     */
    smsUrl: string;
    addressRequirements: DependentPhoneNumberAddressRequirement;
    /**
     * The set of Boolean properties that indicates whether a phone number can receive calls or messages.  Capabilities are  `Voice`, `SMS`, and `MMS` and each capability can be: `true` or `false`.
     */
    capabilities: any;
    /**
     * The URL we call using the `status_callback_method` to send status information to your application.
     */
    statusCallback: string;
    /**
     * The HTTP method we use to call `status_callback`. Can be: `GET` or `POST`.
     */
    statusCallbackMethod: string;
    /**
     * The API version used to start a new TwiML session.
     */
    apiVersion: string;
    /**
     * The SID of the application that handles SMS messages sent to the phone number. If an `sms_application_sid` is present, we ignore all `sms_*_url` values and use those of the application.
     */
    smsApplicationSid: string;
    /**
     * The SID of the application that handles calls to the phone number. If a `voice_application_sid` is present, we ignore all of the voice urls and use those set on the application. Setting a `voice_application_sid` will automatically delete your `trunk_sid` and vice versa.
     */
    voiceApplicationSid: string;
    /**
     * The SID of the Trunk that handles calls to the phone number. If a `trunk_sid` is present, we ignore all of the voice urls and voice applications and use those set on the Trunk. Setting a `trunk_sid` will automatically delete your `voice_application_sid` and vice versa.
     */
    trunkSid: string;
    emergencyStatus: DependentPhoneNumberEmergencyStatus;
    /**
     * The SID of the emergency address configuration that we use for emergency calling from the phone number.
     */
    emergencyAddressSid: string;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        friendlyName: string;
        phoneNumber: string;
        voiceUrl: string;
        voiceMethod: string;
        voiceFallbackMethod: string;
        voiceFallbackUrl: string;
        voiceCallerIdLookup: boolean;
        dateCreated: Date;
        dateUpdated: Date;
        smsFallbackMethod: string;
        smsFallbackUrl: string;
        smsMethod: string;
        smsUrl: string;
        addressRequirements: DependentPhoneNumberAddressRequirement;
        capabilities: any;
        statusCallback: string;
        statusCallbackMethod: string;
        apiVersion: string;
        smsApplicationSid: string;
        voiceApplicationSid: string;
        trunkSid: string;
        emergencyStatus: DependentPhoneNumberEmergencyStatus;
        emergencyAddressSid: string;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class DependentPhoneNumberPage extends Page<V2010, DependentPhoneNumberPayload, DependentPhoneNumberResource, DependentPhoneNumberInstance> {
    /**
     * Initialize the DependentPhoneNumberPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: DependentPhoneNumberSolution);
    /**
     * Build an instance of DependentPhoneNumberInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DependentPhoneNumberResource): DependentPhoneNumberInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
