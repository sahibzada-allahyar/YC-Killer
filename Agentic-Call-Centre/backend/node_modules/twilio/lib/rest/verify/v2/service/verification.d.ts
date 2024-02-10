/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../../V2";
export type VerificationChannel = "sms" | "call" | "email" | "whatsapp" | "sna";
export type VerificationRiskCheck = "enable" | "disable";
export type VerificationStatus = "canceled" | "approved";
/**
 * Options to pass to update a VerificationInstance
 */
export interface VerificationContextUpdateOptions {
    /**  */
    status: VerificationStatus;
}
/**
 * Options to pass to create a VerificationInstance
 */
export interface VerificationListInstanceCreateOptions {
    /** The phone number or [email](https://www.twilio.com/docs/verify/email) to verify. Phone numbers must be in [E.164 format](https://www.twilio.com/docs/glossary/what-e164). */
    to: string;
    /** The verification method to use. One of: [`email`](https://www.twilio.com/docs/verify/email), `sms`, `whatsapp`, `call`, `sna` or `auto`. */
    channel: string;
    /** A custom user defined friendly name that overwrites the existing one in the verification message */
    customFriendlyName?: string;
    /** The text of a custom message to use for the verification. */
    customMessage?: string;
    /** The digits to send after a phone call is answered, for example, to dial an extension. For more information, see the Programmable Voice documentation of [sendDigits](https://www.twilio.com/docs/voice/twiml/number#attributes-sendDigits). */
    sendDigits?: string;
    /** Locale will automatically resolve based on phone number country code for SMS, WhatsApp, and call channel verifications. It will fallback to English or the template’s default translation if the selected translation is not available. This parameter will override the automatic locale resolution. [See supported languages and more information here](https://www.twilio.com/docs/verify/supported-languages). */
    locale?: string;
    /** A pre-generated code to use for verification. The code can be between 4 and 10 characters, inclusive. */
    customCode?: string;
    /** The amount of the associated PSD2 compliant transaction. Requires the PSD2 Service flag enabled. */
    amount?: string;
    /** The payee of the associated PSD2 compliant transaction. Requires the PSD2 Service flag enabled. */
    payee?: string;
    /** The custom key-value pairs of Programmable Rate Limits. Keys correspond to `unique_name` fields defined when [creating your Rate Limit](https://www.twilio.com/docs/verify/api/service-rate-limits). Associated value pairs represent values in the request that you are rate limiting on. You may include multiple Rate Limit values in each request. */
    rateLimits?: any;
    /** [`email`](https://www.twilio.com/docs/verify/email) channel configuration in json format. The fields \\\'from\\\' and \\\'from_name\\\' are optional but if included the \\\'from\\\' field must have a valid email address. */
    channelConfiguration?: any;
    /** Your [App Hash](https://developers.google.com/identity/sms-retriever/verify#computing_your_apps_hash_string) to be appended at the end of your verification SMS body. Applies only to SMS. Example SMS body: `<#> Your AppName verification code is: 1234 He42w354ol9`. */
    appHash?: string;
    /** The message [template](https://www.twilio.com/docs/verify/api/templates). If provided, will override the default template for the Service. SMS and Voice channels only. */
    templateSid?: string;
    /** A stringified JSON object in which the keys are the template\\\'s special variables and the values are the variables substitutions. */
    templateCustomSubstitutions?: string;
    /** Strongly encouraged if using the auto channel. The IP address of the client\\\'s device. If provided, it has to be a valid IPv4 or IPv6 address. */
    deviceIp?: string;
    /**  */
    riskCheck?: VerificationRiskCheck;
    /** A string containing a JSON map of key value pairs of tags to be recorded as metadata for the message. The object may contain up to 10 tags. Keys and values can each be up to 128 characters in length. */
    tags?: string;
}
export interface VerificationContext {
    /**
     * Fetch a VerificationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationInstance
     */
    fetch(callback?: (error: Error | null, item?: VerificationInstance) => any): Promise<VerificationInstance>;
    /**
     * Update a VerificationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationInstance
     */
    update(params: VerificationContextUpdateOptions, callback?: (error: Error | null, item?: VerificationInstance) => any): Promise<VerificationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface VerificationContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class VerificationContextImpl implements VerificationContext {
    protected _version: V2;
    protected _solution: VerificationContextSolution;
    protected _uri: string;
    constructor(_version: V2, serviceSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: VerificationInstance) => any): Promise<VerificationInstance>;
    update(params: VerificationContextUpdateOptions, callback?: (error: Error | null, item?: VerificationInstance) => any): Promise<VerificationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): VerificationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface VerificationResource {
    sid: string;
    service_sid: string;
    account_sid: string;
    to: string;
    channel: VerificationChannel;
    status: string;
    valid: boolean;
    lookup: any;
    amount: string;
    payee: string;
    send_code_attempts: Array<any>;
    date_created: Date;
    date_updated: Date;
    sna: any;
    url: string;
}
export declare class VerificationInstance {
    protected _version: V2;
    protected _solution: VerificationContextSolution;
    protected _context?: VerificationContext;
    constructor(_version: V2, payload: VerificationResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Verification resource.
     */
    sid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/verify/api/service) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Verification resource.
     */
    accountSid: string;
    /**
     * The phone number or [email](https://www.twilio.com/docs/verify/email) being verified. Phone numbers must be in [E.164 format](https://www.twilio.com/docs/glossary/what-e164).
     */
    to: string;
    channel: VerificationChannel;
    /**
     * The status of the verification. One of: `pending`, `approved`, or `canceled`
     */
    status: string;
    /**
     * Use \"status\" instead. Legacy property indicating whether the verification was successful.
     */
    valid: boolean;
    /**
     * Information about the phone number being verified.
     */
    lookup: any;
    /**
     * The amount of the associated PSD2 compliant transaction. Requires the PSD2 Service flag enabled.
     */
    amount: string;
    /**
     * The payee of the associated PSD2 compliant transaction. Requires the PSD2 Service flag enabled.
     */
    payee: string;
    /**
     * An array of verification attempt objects containing the channel attempted and the channel-specific transaction SID.
     */
    sendCodeAttempts: Array<any>;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The set of fields used for a silent network auth (`sna`) verification. Contains a single field with the URL to be invoked to verify the phone number.
     */
    sna: any;
    /**
     * The absolute URL of the Verification resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a VerificationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationInstance
     */
    fetch(callback?: (error: Error | null, item?: VerificationInstance) => any): Promise<VerificationInstance>;
    /**
     * Update a VerificationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationInstance
     */
    update(params: VerificationContextUpdateOptions, callback?: (error: Error | null, item?: VerificationInstance) => any): Promise<VerificationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        serviceSid: string;
        accountSid: string;
        to: string;
        channel: VerificationChannel;
        status: string;
        valid: boolean;
        lookup: any;
        amount: string;
        payee: string;
        sendCodeAttempts: any[];
        dateCreated: Date;
        dateUpdated: Date;
        sna: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface VerificationSolution {
    serviceSid: string;
}
export interface VerificationListInstance {
    _version: V2;
    _solution: VerificationSolution;
    _uri: string;
    (sid: string): VerificationContext;
    get(sid: string): VerificationContext;
    /**
     * Create a VerificationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationInstance
     */
    create(params: VerificationListInstanceCreateOptions, callback?: (error: Error | null, item?: VerificationInstance) => any): Promise<VerificationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function VerificationListInstance(version: V2, serviceSid: string): VerificationListInstance;
export {};
