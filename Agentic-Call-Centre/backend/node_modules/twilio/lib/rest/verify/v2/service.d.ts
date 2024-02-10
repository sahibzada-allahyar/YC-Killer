/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V2 from "../V2";
import { AccessTokenListInstance } from "./service/accessToken";
import { EntityListInstance } from "./service/entity";
import { MessagingConfigurationListInstance } from "./service/messagingConfiguration";
import { RateLimitListInstance } from "./service/rateLimit";
import { VerificationListInstance } from "./service/verification";
import { VerificationCheckListInstance } from "./service/verificationCheck";
import { WebhookListInstance } from "./service/webhook";
/**
 * Options to pass to update a ServiceInstance
 */
export interface ServiceContextUpdateOptions {
    /** A descriptive string that you create to describe the verification service. It can be up to 32 characters long. **This value should not contain PII.** */
    friendlyName?: string;
    /** The length of the verification code to generate. Must be an integer value between 4 and 10, inclusive. */
    codeLength?: number;
    /** Whether to perform a lookup with each verification started and return info about the phone number. */
    lookupEnabled?: boolean;
    /** Whether to skip sending SMS verifications to landlines. Requires `lookup_enabled`. */
    skipSmsToLandlines?: boolean;
    /** Whether to ask the user to press a number before delivering the verify code in a phone call. */
    dtmfInputRequired?: boolean;
    /** The name of an alternative text-to-speech service to use in phone calls. Applies only to TTS languages. */
    ttsName?: string;
    /** Whether to pass PSD2 transaction parameters when starting a verification. */
    psd2Enabled?: boolean;
    /** Whether to add a privacy warning at the end of an SMS. **Disabled by default and applies only for SMS.** */
    doNotShareWarningEnabled?: boolean;
    /** Whether to allow sending verifications with a custom code instead of a randomly generated one. Not available for all customers. */
    customCodeEnabled?: boolean;
    /** Optional configuration for the Push factors. If true, include the date in the Challenge\\\'s response. Otherwise, the date is omitted from the response. See [Challenge](https://www.twilio.com/docs/verify/api/challenge) resource’s details parameter for more info. Default: false. **Deprecated** do not use this parameter. */
    "push.includeDate"?: boolean;
    /** Optional configuration for the Push factors. Set the APN Credential for this service. This will allow to send push notifications to iOS devices. See [Credential Resource](https://www.twilio.com/docs/notify/api/credential-resource) */
    "push.apnCredentialSid"?: string;
    /** Optional configuration for the Push factors. Set the FCM Credential for this service. This will allow to send push notifications to Android devices. See [Credential Resource](https://www.twilio.com/docs/notify/api/credential-resource) */
    "push.fcmCredentialSid"?: string;
    /** Optional configuration for the TOTP factors. Set TOTP Issuer for this service. This will allow to configure the issuer of the TOTP URI. */
    "totp.issuer"?: string;
    /** Optional configuration for the TOTP factors. Defines how often, in seconds, are TOTP codes generated. i.e, a new TOTP code is generated every time_step seconds. Must be between 20 and 60 seconds, inclusive. Defaults to 30 seconds */
    "totp.timeStep"?: number;
    /** Optional configuration for the TOTP factors. Number of digits for generated TOTP codes. Must be between 3 and 8, inclusive. Defaults to 6 */
    "totp.codeLength"?: number;
    /** Optional configuration for the TOTP factors. The number of time-steps, past and future, that are valid for validation of TOTP codes. Must be between 0 and 2, inclusive. Defaults to 1 */
    "totp.skew"?: number;
    /** The default message [template](https://www.twilio.com/docs/verify/api/templates). Will be used for all SMS verifications unless explicitly overriden. SMS channel only. */
    defaultTemplateSid?: string;
    /** Whether to allow verifications from the service to reach the stream-events sinks if configured */
    verifyEventSubscriptionEnabled?: boolean;
}
/**
 * Options to pass to create a ServiceInstance
 */
export interface ServiceListInstanceCreateOptions {
    /** A descriptive string that you create to describe the verification service. It can be up to 32 characters long. **This value should not contain PII.** */
    friendlyName: string;
    /** The length of the verification code to generate. Must be an integer value between 4 and 10, inclusive. */
    codeLength?: number;
    /** Whether to perform a lookup with each verification started and return info about the phone number. */
    lookupEnabled?: boolean;
    /** Whether to skip sending SMS verifications to landlines. Requires `lookup_enabled`. */
    skipSmsToLandlines?: boolean;
    /** Whether to ask the user to press a number before delivering the verify code in a phone call. */
    dtmfInputRequired?: boolean;
    /** The name of an alternative text-to-speech service to use in phone calls. Applies only to TTS languages. */
    ttsName?: string;
    /** Whether to pass PSD2 transaction parameters when starting a verification. */
    psd2Enabled?: boolean;
    /** Whether to add a security warning at the end of an SMS verification body. Disabled by default and applies only to SMS. Example SMS body: `Your AppName verification code is: 1234. Don’t share this code with anyone; our employees will never ask for the code` */
    doNotShareWarningEnabled?: boolean;
    /** Whether to allow sending verifications with a custom code instead of a randomly generated one. Not available for all customers. */
    customCodeEnabled?: boolean;
    /** Optional configuration for the Push factors. If true, include the date in the Challenge\\\'s response. Otherwise, the date is omitted from the response. See [Challenge](https://www.twilio.com/docs/verify/api/challenge) resource’s details parameter for more info. Default: false. **Deprecated** do not use this parameter. This timestamp value is the same one as the one found in `date_created`, please use that one instead. */
    "push.includeDate"?: boolean;
    /** Optional configuration for the Push factors. Set the APN Credential for this service. This will allow to send push notifications to iOS devices. See [Credential Resource](https://www.twilio.com/docs/notify/api/credential-resource) */
    "push.apnCredentialSid"?: string;
    /** Optional configuration for the Push factors. Set the FCM Credential for this service. This will allow to send push notifications to Android devices. See [Credential Resource](https://www.twilio.com/docs/notify/api/credential-resource) */
    "push.fcmCredentialSid"?: string;
    /** Optional configuration for the TOTP factors. Set TOTP Issuer for this service. This will allow to configure the issuer of the TOTP URI. Defaults to the service friendly name if not provided. */
    "totp.issuer"?: string;
    /** Optional configuration for the TOTP factors. Defines how often, in seconds, are TOTP codes generated. i.e, a new TOTP code is generated every time_step seconds. Must be between 20 and 60 seconds, inclusive. Defaults to 30 seconds */
    "totp.timeStep"?: number;
    /** Optional configuration for the TOTP factors. Number of digits for generated TOTP codes. Must be between 3 and 8, inclusive. Defaults to 6 */
    "totp.codeLength"?: number;
    /** Optional configuration for the TOTP factors. The number of time-steps, past and future, that are valid for validation of TOTP codes. Must be between 0 and 2, inclusive. Defaults to 1 */
    "totp.skew"?: number;
    /** The default message [template](https://www.twilio.com/docs/verify/api/templates). Will be used for all SMS verifications unless explicitly overriden. SMS channel only. */
    defaultTemplateSid?: string;
    /** Whether to allow verifications from the service to reach the stream-events sinks if configured */
    verifyEventSubscriptionEnabled?: boolean;
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
    accessTokens: AccessTokenListInstance;
    entities: EntityListInstance;
    messagingConfigurations: MessagingConfigurationListInstance;
    rateLimits: RateLimitListInstance;
    verifications: VerificationListInstance;
    verificationChecks: VerificationCheckListInstance;
    webhooks: WebhookListInstance;
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
    protected _version: V2;
    protected _solution: ServiceContextSolution;
    protected _uri: string;
    protected _accessTokens?: AccessTokenListInstance;
    protected _entities?: EntityListInstance;
    protected _messagingConfigurations?: MessagingConfigurationListInstance;
    protected _rateLimits?: RateLimitListInstance;
    protected _verifications?: VerificationListInstance;
    protected _verificationChecks?: VerificationCheckListInstance;
    protected _webhooks?: WebhookListInstance;
    constructor(_version: V2, sid: string);
    get accessTokens(): AccessTokenListInstance;
    get entities(): EntityListInstance;
    get messagingConfigurations(): MessagingConfigurationListInstance;
    get rateLimits(): RateLimitListInstance;
    get verifications(): VerificationListInstance;
    get verificationChecks(): VerificationCheckListInstance;
    get webhooks(): WebhookListInstance;
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
    code_length: number;
    lookup_enabled: boolean;
    psd2_enabled: boolean;
    skip_sms_to_landlines: boolean;
    dtmf_input_required: boolean;
    tts_name: string;
    do_not_share_warning_enabled: boolean;
    custom_code_enabled: boolean;
    push: any;
    totp: any;
    default_template_sid: string;
    verify_event_subscription_enabled: boolean;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class ServiceInstance {
    protected _version: V2;
    protected _solution: ServiceContextSolution;
    protected _context?: ServiceContext;
    constructor(_version: V2, payload: ServiceResource, sid?: string);
    /**
     * The unique string that we created to identify the Service resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Service resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the verification service. It can be up to 32 characters long. **This value should not contain PII.**
     */
    friendlyName: string;
    /**
     * The length of the verification code to generate.
     */
    codeLength: number;
    /**
     * Whether to perform a lookup with each verification started and return info about the phone number.
     */
    lookupEnabled: boolean;
    /**
     * Whether to pass PSD2 transaction parameters when starting a verification.
     */
    psd2Enabled: boolean;
    /**
     * Whether to skip sending SMS verifications to landlines. Requires `lookup_enabled`.
     */
    skipSmsToLandlines: boolean;
    /**
     * Whether to ask the user to press a number before delivering the verify code in a phone call.
     */
    dtmfInputRequired: boolean;
    /**
     * The name of an alternative text-to-speech service to use in phone calls. Applies only to TTS languages.
     */
    ttsName: string;
    /**
     * Whether to add a security warning at the end of an SMS verification body. Disabled by default and applies only to SMS. Example SMS body: `Your AppName verification code is: 1234. Don’t share this code with anyone; our employees will never ask for the code`
     */
    doNotShareWarningEnabled: boolean;
    /**
     * Whether to allow sending verifications with a custom code instead of a randomly generated one. Not available for all customers.
     */
    customCodeEnabled: boolean;
    /**
     * Configurations for the Push factors (channel) created under this Service.
     */
    push: any;
    /**
     * Configurations for the TOTP factors (channel) created under this Service.
     */
    totp: any;
    defaultTemplateSid: string;
    /**
     * Whether to allow verifications from the service to reach the stream-events sinks if configured
     */
    verifyEventSubscriptionEnabled: boolean;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    /**
     * The URLs of related resources.
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
     * Access the accessTokens.
     */
    accessTokens(): AccessTokenListInstance;
    /**
     * Access the entities.
     */
    entities(): EntityListInstance;
    /**
     * Access the messagingConfigurations.
     */
    messagingConfigurations(): MessagingConfigurationListInstance;
    /**
     * Access the rateLimits.
     */
    rateLimits(): RateLimitListInstance;
    /**
     * Access the verifications.
     */
    verifications(): VerificationListInstance;
    /**
     * Access the verificationChecks.
     */
    verificationChecks(): VerificationCheckListInstance;
    /**
     * Access the webhooks.
     */
    webhooks(): WebhookListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        friendlyName: string;
        codeLength: number;
        lookupEnabled: boolean;
        psd2Enabled: boolean;
        skipSmsToLandlines: boolean;
        dtmfInputRequired: boolean;
        ttsName: string;
        doNotShareWarningEnabled: boolean;
        customCodeEnabled: boolean;
        push: any;
        totp: any;
        defaultTemplateSid: string;
        verifyEventSubscriptionEnabled: boolean;
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
    _version: V2;
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
export declare function ServiceListInstance(version: V2): ServiceListInstance;
export declare class ServicePage extends Page<V2, ServicePayload, ServiceResource, ServiceInstance> {
    /**
     * Initialize the ServicePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: ServiceSolution);
    /**
     * Build an instance of ServiceInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ServiceResource): ServiceInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
