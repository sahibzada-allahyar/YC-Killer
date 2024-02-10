/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
import { AssignedAddOnListInstance } from "./incomingPhoneNumber/assignedAddOn";
import { LocalListInstance } from "./incomingPhoneNumber/local";
import { MobileListInstance } from "./incomingPhoneNumber/mobile";
import { TollFreeListInstance } from "./incomingPhoneNumber/tollFree";
import { PhoneNumberCapabilities } from "../../../../interfaces";
export type IncomingPhoneNumberAddressRequirement = "none" | "any" | "local" | "foreign";
export type IncomingPhoneNumberEmergencyAddressStatus = "registered" | "unregistered" | "pending-registration" | "registration-failure" | "pending-unregistration" | "unregistration-failure";
export type IncomingPhoneNumberEmergencyStatus = "Active" | "Inactive";
export type IncomingPhoneNumberVoiceReceiveMode = "voice" | "fax";
/**
 * Options to pass to update a IncomingPhoneNumberInstance
 */
export interface IncomingPhoneNumberContextUpdateOptions {
    /** The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the IncomingPhoneNumber resource to update.  For more information, see [Exchanging Numbers Between Subaccounts](https://www.twilio.com/docs/iam/api/subaccounts#exchanging-numbers). */
    accountSid?: string;
    /** The API version to use for incoming calls made to the phone number. The default is `2010-04-01`. */
    apiVersion?: string;
    /** A descriptive string that you created to describe this phone number. It can be up to 64 characters long. By default, this is a formatted version of the phone number. */
    friendlyName?: string;
    /** The SID of the application that should handle SMS messages sent to the number. If an `sms_application_sid` is present, we ignore all of the `sms_*_url` urls and use those set on the application. */
    smsApplicationSid?: string;
    /** The HTTP method that we should use to call `sms_fallback_url`. Can be: `GET` or `POST` and defaults to `POST`. */
    smsFallbackMethod?: string;
    /** The URL that we should call when an error occurs while requesting or executing the TwiML defined by `sms_url`. */
    smsFallbackUrl?: string;
    /** The HTTP method that we should use to call `sms_url`. Can be: `GET` or `POST` and defaults to `POST`. */
    smsMethod?: string;
    /** The URL we should call when the phone number receives an incoming SMS message. */
    smsUrl?: string;
    /** The URL we should call using the `status_callback_method` to send status information to your application. */
    statusCallback?: string;
    /** The HTTP method we should use to call `status_callback`. Can be: `GET` or `POST` and defaults to `POST`. */
    statusCallbackMethod?: string;
    /** The SID of the application we should use to handle phone calls to the phone number. If a `voice_application_sid` is present, we ignore all of the voice urls and use only those set on the application. Setting a `voice_application_sid` will automatically delete your `trunk_sid` and vice versa. */
    voiceApplicationSid?: string;
    /** Whether to lookup the caller\\\'s name from the CNAM database and post it to your app. Can be: `true` or `false` and defaults to `false`. */
    voiceCallerIdLookup?: boolean;
    /** The HTTP method that we should use to call `voice_fallback_url`. Can be: `GET` or `POST` and defaults to `POST`. */
    voiceFallbackMethod?: string;
    /** The URL that we should call when an error occurs retrieving or executing the TwiML requested by `url`. */
    voiceFallbackUrl?: string;
    /** The HTTP method that we should use to call `voice_url`. Can be: `GET` or `POST` and defaults to `POST`. */
    voiceMethod?: string;
    /** The URL that we should call to answer a call to the phone number. The `voice_url` will not be called if a `voice_application_sid` or a `trunk_sid` is set. */
    voiceUrl?: string;
    /**  */
    emergencyStatus?: IncomingPhoneNumberEmergencyStatus;
    /** The SID of the emergency address configuration to use for emergency calling from this phone number. */
    emergencyAddressSid?: string;
    /** The SID of the Trunk we should use to handle phone calls to the phone number. If a `trunk_sid` is present, we ignore all of the voice urls and voice applications and use only those set on the Trunk. Setting a `trunk_sid` will automatically delete your `voice_application_sid` and vice versa. */
    trunkSid?: string;
    /**  */
    voiceReceiveMode?: IncomingPhoneNumberVoiceReceiveMode;
    /** The SID of the Identity resource that we should associate with the phone number. Some regions require an identity to meet local regulations. */
    identitySid?: string;
    /** The SID of the Address resource we should associate with the phone number. Some regions require addresses to meet local regulations. */
    addressSid?: string;
    /** The SID of the Bundle resource that you associate with the phone number. Some regions require a Bundle to meet local Regulations. */
    bundleSid?: string;
}
/**
 * Options to pass to create a IncomingPhoneNumberInstance
 */
export interface IncomingPhoneNumberListInstanceCreateOptions {
    /** The API version to use for incoming calls made to the new phone number. The default is `2010-04-01`. */
    apiVersion?: string;
    /** A descriptive string that you created to describe the new phone number. It can be up to 64 characters long. By default, this is a formatted version of the new phone number. */
    friendlyName?: string;
    /** The SID of the application that should handle SMS messages sent to the new phone number. If an `sms_application_sid` is present, we ignore all of the `sms_*_url` urls and use those set on the application. */
    smsApplicationSid?: string;
    /** The HTTP method that we should use to call `sms_fallback_url`. Can be: `GET` or `POST` and defaults to `POST`. */
    smsFallbackMethod?: string;
    /** The URL that we should call when an error occurs while requesting or executing the TwiML defined by `sms_url`. */
    smsFallbackUrl?: string;
    /** The HTTP method that we should use to call `sms_url`. Can be: `GET` or `POST` and defaults to `POST`. */
    smsMethod?: string;
    /** The URL we should call when the new phone number receives an incoming SMS message. */
    smsUrl?: string;
    /** The URL we should call using the `status_callback_method` to send status information to your application. */
    statusCallback?: string;
    /** The HTTP method we should use to call `status_callback`. Can be: `GET` or `POST` and defaults to `POST`. */
    statusCallbackMethod?: string;
    /** The SID of the application we should use to handle calls to the new phone number. If a `voice_application_sid` is present, we ignore all of the voice urls and use only those set on the application. Setting a `voice_application_sid` will automatically delete your `trunk_sid` and vice versa. */
    voiceApplicationSid?: string;
    /** Whether to lookup the caller\\\'s name from the CNAM database and post it to your app. Can be: `true` or `false` and defaults to `false`. */
    voiceCallerIdLookup?: boolean;
    /** The HTTP method that we should use to call `voice_fallback_url`. Can be: `GET` or `POST` and defaults to `POST`. */
    voiceFallbackMethod?: string;
    /** The URL that we should call when an error occurs retrieving or executing the TwiML requested by `url`. */
    voiceFallbackUrl?: string;
    /** The HTTP method that we should use to call `voice_url`. Can be: `GET` or `POST` and defaults to `POST`. */
    voiceMethod?: string;
    /** The URL that we should call to answer a call to the new phone number. The `voice_url` will not be called if a `voice_application_sid` or a `trunk_sid` is set. */
    voiceUrl?: string;
    /**  */
    emergencyStatus?: IncomingPhoneNumberEmergencyStatus;
    /** The SID of the emergency address configuration to use for emergency calling from the new phone number. */
    emergencyAddressSid?: string;
    /** The SID of the Trunk we should use to handle calls to the new phone number. If a `trunk_sid` is present, we ignore all of the voice urls and voice applications and use only those set on the Trunk. Setting a `trunk_sid` will automatically delete your `voice_application_sid` and vice versa. */
    trunkSid?: string;
    /** The SID of the Identity resource that we should associate with the new phone number. Some regions require an identity to meet local regulations. */
    identitySid?: string;
    /** The SID of the Address resource we should associate with the new phone number. Some regions require addresses to meet local regulations. */
    addressSid?: string;
    /**  */
    voiceReceiveMode?: IncomingPhoneNumberVoiceReceiveMode;
    /** The SID of the Bundle resource that you associate with the phone number. Some regions require a Bundle to meet local Regulations. */
    bundleSid?: string;
    /** The phone number to purchase specified in [E.164](https://www.twilio.com/docs/glossary/what-e164) format.  E.164 phone numbers consist of a + followed by the country code and subscriber number without punctuation characters. For example, +14155551234. */
    phoneNumber?: string;
    /** The desired area code for your new incoming phone number. Can be any three-digit, US or Canada area code. We will provision an available phone number within this area code for you. **You must provide an `area_code` or a `phone_number`.** (US and Canada only). */
    areaCode?: string;
}
/**
 * Options to pass to each
 */
export interface IncomingPhoneNumberListInstanceEachOptions {
    /** Whether to include phone numbers new to the Twilio platform. Can be: `true` or `false` and the default is `true`. */
    beta?: boolean;
    /** A string that identifies the IncomingPhoneNumber resources to read. */
    friendlyName?: string;
    /** The phone numbers of the IncomingPhoneNumber resources to read. You can specify partial numbers and use \'*\' as a wildcard for any digit. */
    phoneNumber?: string;
    /** Whether to include phone numbers based on their origin. Can be: `twilio` or `hosted`. By default, phone numbers of all origin are included. */
    origin?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: IncomingPhoneNumberInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface IncomingPhoneNumberListInstanceOptions {
    /** Whether to include phone numbers new to the Twilio platform. Can be: `true` or `false` and the default is `true`. */
    beta?: boolean;
    /** A string that identifies the IncomingPhoneNumber resources to read. */
    friendlyName?: string;
    /** The phone numbers of the IncomingPhoneNumber resources to read. You can specify partial numbers and use \'*\' as a wildcard for any digit. */
    phoneNumber?: string;
    /** Whether to include phone numbers based on their origin. Can be: `twilio` or `hosted`. By default, phone numbers of all origin are included. */
    origin?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface IncomingPhoneNumberListInstancePageOptions {
    /** Whether to include phone numbers new to the Twilio platform. Can be: `true` or `false` and the default is `true`. */
    beta?: boolean;
    /** A string that identifies the IncomingPhoneNumber resources to read. */
    friendlyName?: string;
    /** The phone numbers of the IncomingPhoneNumber resources to read. You can specify partial numbers and use \'*\' as a wildcard for any digit. */
    phoneNumber?: string;
    /** Whether to include phone numbers based on their origin. Can be: `twilio` or `hosted`. By default, phone numbers of all origin are included. */
    origin?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface IncomingPhoneNumberContext {
    assignedAddOns: AssignedAddOnListInstance;
    /**
     * Remove a IncomingPhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a IncomingPhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IncomingPhoneNumberInstance
     */
    fetch(callback?: (error: Error | null, item?: IncomingPhoneNumberInstance) => any): Promise<IncomingPhoneNumberInstance>;
    /**
     * Update a IncomingPhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IncomingPhoneNumberInstance
     */
    update(callback?: (error: Error | null, item?: IncomingPhoneNumberInstance) => any): Promise<IncomingPhoneNumberInstance>;
    /**
     * Update a IncomingPhoneNumberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IncomingPhoneNumberInstance
     */
    update(params: IncomingPhoneNumberContextUpdateOptions, callback?: (error: Error | null, item?: IncomingPhoneNumberInstance) => any): Promise<IncomingPhoneNumberInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface IncomingPhoneNumberContextSolution {
    accountSid: string;
    sid: string;
}
export declare class IncomingPhoneNumberContextImpl implements IncomingPhoneNumberContext {
    protected _version: V2010;
    protected _solution: IncomingPhoneNumberContextSolution;
    protected _uri: string;
    protected _assignedAddOns?: AssignedAddOnListInstance;
    constructor(_version: V2010, accountSid: string, sid: string);
    get assignedAddOns(): AssignedAddOnListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: IncomingPhoneNumberInstance) => any): Promise<IncomingPhoneNumberInstance>;
    update(params?: IncomingPhoneNumberContextUpdateOptions | ((error: Error | null, item?: IncomingPhoneNumberInstance) => any), callback?: (error: Error | null, item?: IncomingPhoneNumberInstance) => any): Promise<IncomingPhoneNumberInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): IncomingPhoneNumberContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface IncomingPhoneNumberPayload extends TwilioResponsePayload {
    incoming_phone_numbers: IncomingPhoneNumberResource[];
}
interface IncomingPhoneNumberResource {
    account_sid: string;
    address_sid: string;
    address_requirements: IncomingPhoneNumberAddressRequirement;
    api_version: string;
    beta: boolean;
    capabilities: PhoneNumberCapabilities;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    identity_sid: string;
    phone_number: string;
    origin: string;
    sid: string;
    sms_application_sid: string;
    sms_fallback_method: string;
    sms_fallback_url: string;
    sms_method: string;
    sms_url: string;
    status_callback: string;
    status_callback_method: string;
    trunk_sid: string;
    uri: string;
    voice_receive_mode: IncomingPhoneNumberVoiceReceiveMode;
    voice_application_sid: string;
    voice_caller_id_lookup: boolean;
    voice_fallback_method: string;
    voice_fallback_url: string;
    voice_method: string;
    voice_url: string;
    emergency_status: IncomingPhoneNumberEmergencyStatus;
    emergency_address_sid: string;
    emergency_address_status: IncomingPhoneNumberEmergencyAddressStatus;
    bundle_sid: string;
    status: string;
}
export declare class IncomingPhoneNumberInstance {
    protected _version: V2010;
    protected _solution: IncomingPhoneNumberContextSolution;
    protected _context?: IncomingPhoneNumberContext;
    constructor(_version: V2010, payload: IncomingPhoneNumberResource, accountSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created this IncomingPhoneNumber resource.
     */
    accountSid: string;
    /**
     * The SID of the Address resource associated with the phone number.
     */
    addressSid: string;
    addressRequirements: IncomingPhoneNumberAddressRequirement;
    /**
     * The API version used to start a new TwiML session.
     */
    apiVersion: string;
    /**
     * Whether the phone number is new to the Twilio platform. Can be: `true` or `false`.
     */
    beta: boolean;
    capabilities: PhoneNumberCapabilities;
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
     * The SID of the Identity resource that we associate with the phone number. Some regions require an Identity to meet local regulations.
     */
    identitySid: string;
    /**
     * The phone number in [E.164](https://www.twilio.com/docs/glossary/what-e164) format, which consists of a + followed by the country code and subscriber number.
     */
    phoneNumber: string;
    /**
     * The phone number\'s origin. `twilio` identifies Twilio-owned phone numbers and `hosted` identifies hosted phone numbers.
     */
    origin: string;
    /**
     * The unique string that that we created to identify this IncomingPhoneNumber resource.
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
    /**
     * The URL we call using the `status_callback_method` to send status information to your application.
     */
    statusCallback: string;
    /**
     * The HTTP method we use to call `status_callback`. Can be: `GET` or `POST`.
     */
    statusCallbackMethod: string;
    /**
     * The SID of the Trunk that handles calls to the phone number. If a `trunk_sid` is present, we ignore all of the voice urls and voice applications and use those set on the Trunk. Setting a `trunk_sid` will automatically delete your `voice_application_sid` and vice versa.
     */
    trunkSid: string;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    voiceReceiveMode: IncomingPhoneNumberVoiceReceiveMode;
    /**
     * The SID of the application that handles calls to the phone number. If a `voice_application_sid` is present, we ignore all of the voice urls and use those set on the application. Setting a `voice_application_sid` will automatically delete your `trunk_sid` and vice versa.
     */
    voiceApplicationSid: string;
    /**
     * Whether we look up the caller\'s caller-ID name from the CNAM database ($0.01 per look up). Can be: `true` or `false`.
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
     * The URL we call when the phone number receives a call. The `voice_url` will not be used if a `voice_application_sid` or a `trunk_sid` is set.
     */
    voiceUrl: string;
    emergencyStatus: IncomingPhoneNumberEmergencyStatus;
    /**
     * The SID of the emergency address configuration that we use for emergency calling from this phone number.
     */
    emergencyAddressSid: string;
    emergencyAddressStatus: IncomingPhoneNumberEmergencyAddressStatus;
    /**
     * The SID of the Bundle resource that you associate with the phone number. Some regions require a Bundle to meet local Regulations.
     */
    bundleSid: string;
    status: string;
    private get _proxy();
    /**
     * Remove a IncomingPhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a IncomingPhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IncomingPhoneNumberInstance
     */
    fetch(callback?: (error: Error | null, item?: IncomingPhoneNumberInstance) => any): Promise<IncomingPhoneNumberInstance>;
    /**
     * Update a IncomingPhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IncomingPhoneNumberInstance
     */
    update(callback?: (error: Error | null, item?: IncomingPhoneNumberInstance) => any): Promise<IncomingPhoneNumberInstance>;
    /**
     * Update a IncomingPhoneNumberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IncomingPhoneNumberInstance
     */
    update(params: IncomingPhoneNumberContextUpdateOptions, callback?: (error: Error | null, item?: IncomingPhoneNumberInstance) => any): Promise<IncomingPhoneNumberInstance>;
    /**
     * Access the assignedAddOns.
     */
    assignedAddOns(): AssignedAddOnListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        addressSid: string;
        addressRequirements: IncomingPhoneNumberAddressRequirement;
        apiVersion: string;
        beta: boolean;
        capabilities: PhoneNumberCapabilities;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        identitySid: string;
        phoneNumber: string;
        origin: string;
        sid: string;
        smsApplicationSid: string;
        smsFallbackMethod: string;
        smsFallbackUrl: string;
        smsMethod: string;
        smsUrl: string;
        statusCallback: string;
        statusCallbackMethod: string;
        trunkSid: string;
        uri: string;
        voiceReceiveMode: IncomingPhoneNumberVoiceReceiveMode;
        voiceApplicationSid: string;
        voiceCallerIdLookup: boolean;
        voiceFallbackMethod: string;
        voiceFallbackUrl: string;
        voiceMethod: string;
        voiceUrl: string;
        emergencyStatus: IncomingPhoneNumberEmergencyStatus;
        emergencyAddressSid: string;
        emergencyAddressStatus: IncomingPhoneNumberEmergencyAddressStatus;
        bundleSid: string;
        status: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface IncomingPhoneNumberSolution {
    accountSid: string;
}
export interface IncomingPhoneNumberListInstance {
    _version: V2010;
    _solution: IncomingPhoneNumberSolution;
    _uri: string;
    (sid: string): IncomingPhoneNumberContext;
    get(sid: string): IncomingPhoneNumberContext;
    _local?: LocalListInstance;
    local: LocalListInstance;
    _mobile?: MobileListInstance;
    mobile: MobileListInstance;
    _tollFree?: TollFreeListInstance;
    tollFree: TollFreeListInstance;
    /**
     * Create a IncomingPhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IncomingPhoneNumberInstance
     */
    create(callback?: (error: Error | null, item?: IncomingPhoneNumberInstance) => any): Promise<IncomingPhoneNumberInstance>;
    /**
     * Create a IncomingPhoneNumberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IncomingPhoneNumberInstance
     */
    create(params: IncomingPhoneNumberListInstanceCreateOptions, callback?: (error: Error | null, item?: IncomingPhoneNumberInstance) => any): Promise<IncomingPhoneNumberInstance>;
    /**
     * Streams IncomingPhoneNumberInstance records from the API.
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
     * @param { IncomingPhoneNumberListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: IncomingPhoneNumberInstance, done: (err?: Error) => void) => void): void;
    each(params: IncomingPhoneNumberListInstanceEachOptions, callback?: (item: IncomingPhoneNumberInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of IncomingPhoneNumberInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: IncomingPhoneNumberPage) => any): Promise<IncomingPhoneNumberPage>;
    /**
     * Lists IncomingPhoneNumberInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { IncomingPhoneNumberListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: IncomingPhoneNumberInstance[]) => any): Promise<IncomingPhoneNumberInstance[]>;
    list(params: IncomingPhoneNumberListInstanceOptions, callback?: (error: Error | null, items: IncomingPhoneNumberInstance[]) => any): Promise<IncomingPhoneNumberInstance[]>;
    /**
     * Retrieve a single page of IncomingPhoneNumberInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { IncomingPhoneNumberListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: IncomingPhoneNumberPage) => any): Promise<IncomingPhoneNumberPage>;
    page(params: IncomingPhoneNumberListInstancePageOptions, callback?: (error: Error | null, items: IncomingPhoneNumberPage) => any): Promise<IncomingPhoneNumberPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function IncomingPhoneNumberListInstance(version: V2010, accountSid: string): IncomingPhoneNumberListInstance;
export declare class IncomingPhoneNumberPage extends Page<V2010, IncomingPhoneNumberPayload, IncomingPhoneNumberResource, IncomingPhoneNumberInstance> {
    /**
     * Initialize the IncomingPhoneNumberPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: IncomingPhoneNumberSolution);
    /**
     * Build an instance of IncomingPhoneNumberInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: IncomingPhoneNumberResource): IncomingPhoneNumberInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
