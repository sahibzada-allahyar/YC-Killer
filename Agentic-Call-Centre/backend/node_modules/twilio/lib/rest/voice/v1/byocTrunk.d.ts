/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to update a ByocTrunkInstance
 */
export interface ByocTrunkContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It is not unique and can be up to 255 characters long. */
    friendlyName?: string;
    /** The URL we should call when the BYOC Trunk receives a call. */
    voiceUrl?: string;
    /** The HTTP method we should use to call `voice_url` */
    voiceMethod?: string;
    /** The URL that we should call when an error occurs while retrieving or executing the TwiML requested by `voice_url`. */
    voiceFallbackUrl?: string;
    /** The HTTP method we should use to call `voice_fallback_url`. Can be: `GET` or `POST`. */
    voiceFallbackMethod?: string;
    /** The URL that we should call to pass status parameters (such as call ended) to your application. */
    statusCallbackUrl?: string;
    /** The HTTP method we should use to call `status_callback_url`. Can be: `GET` or `POST`. */
    statusCallbackMethod?: string;
    /** Whether Caller ID Name (CNAM) lookup is enabled for the trunk. If enabled, all inbound calls to the BYOC Trunk from the United States and Canada automatically perform a CNAM Lookup and display Caller ID data on your phone. See [CNAM Lookups](https://www.twilio.com/docs/sip-trunking#CNAM) for more information. */
    cnamLookupEnabled?: boolean;
    /** The SID of the Connection Policy that Twilio will use when routing traffic to your communications infrastructure. */
    connectionPolicySid?: string;
    /** The SID of the SIP Domain that should be used in the `From` header of originating calls sent to your SIP infrastructure. If your SIP infrastructure allows users to \\\"call back\\\" an incoming call, configure this with a [SIP Domain](https://www.twilio.com/docs/voice/api/sending-sip) to ensure proper routing. If not configured, the from domain will default to \\\"sip.twilio.com\\\". */
    fromDomainSid?: string;
}
/**
 * Options to pass to create a ByocTrunkInstance
 */
export interface ByocTrunkListInstanceCreateOptions {
    /** A descriptive string that you create to describe the resource. It is not unique and can be up to 255 characters long. */
    friendlyName?: string;
    /** The URL we should call when the BYOC Trunk receives a call. */
    voiceUrl?: string;
    /** The HTTP method we should use to call `voice_url`. Can be: `GET` or `POST`. */
    voiceMethod?: string;
    /** The URL that we should call when an error occurs while retrieving or executing the TwiML from `voice_url`. */
    voiceFallbackUrl?: string;
    /** The HTTP method we should use to call `voice_fallback_url`. Can be: `GET` or `POST`. */
    voiceFallbackMethod?: string;
    /** The URL that we should call to pass status parameters (such as call ended) to your application. */
    statusCallbackUrl?: string;
    /** The HTTP method we should use to call `status_callback_url`. Can be: `GET` or `POST`. */
    statusCallbackMethod?: string;
    /** Whether Caller ID Name (CNAM) lookup is enabled for the trunk. If enabled, all inbound calls to the BYOC Trunk from the United States and Canada automatically perform a CNAM Lookup and display Caller ID data on your phone. See [CNAM Lookups](https://www.twilio.com/docs/sip-trunking#CNAM) for more information. */
    cnamLookupEnabled?: boolean;
    /** The SID of the Connection Policy that Twilio will use when routing traffic to your communications infrastructure. */
    connectionPolicySid?: string;
    /** The SID of the SIP Domain that should be used in the `From` header of originating calls sent to your SIP infrastructure. If your SIP infrastructure allows users to \\\"call back\\\" an incoming call, configure this with a [SIP Domain](https://www.twilio.com/docs/voice/api/sending-sip) to ensure proper routing. If not configured, the from domain will default to \\\"sip.twilio.com\\\". */
    fromDomainSid?: string;
}
/**
 * Options to pass to each
 */
export interface ByocTrunkListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ByocTrunkInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ByocTrunkListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ByocTrunkListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ByocTrunkContext {
    /**
     * Remove a ByocTrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ByocTrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ByocTrunkInstance
     */
    fetch(callback?: (error: Error | null, item?: ByocTrunkInstance) => any): Promise<ByocTrunkInstance>;
    /**
     * Update a ByocTrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ByocTrunkInstance
     */
    update(callback?: (error: Error | null, item?: ByocTrunkInstance) => any): Promise<ByocTrunkInstance>;
    /**
     * Update a ByocTrunkInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ByocTrunkInstance
     */
    update(params: ByocTrunkContextUpdateOptions, callback?: (error: Error | null, item?: ByocTrunkInstance) => any): Promise<ByocTrunkInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ByocTrunkContextSolution {
    sid: string;
}
export declare class ByocTrunkContextImpl implements ByocTrunkContext {
    protected _version: V1;
    protected _solution: ByocTrunkContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ByocTrunkInstance) => any): Promise<ByocTrunkInstance>;
    update(params?: ByocTrunkContextUpdateOptions | ((error: Error | null, item?: ByocTrunkInstance) => any), callback?: (error: Error | null, item?: ByocTrunkInstance) => any): Promise<ByocTrunkInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ByocTrunkContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ByocTrunkPayload extends TwilioResponsePayload {
    byoc_trunks: ByocTrunkResource[];
}
interface ByocTrunkResource {
    account_sid: string;
    sid: string;
    friendly_name: string;
    voice_url: string;
    voice_method: string;
    voice_fallback_url: string;
    voice_fallback_method: string;
    status_callback_url: string;
    status_callback_method: string;
    cnam_lookup_enabled: boolean;
    connection_policy_sid: string;
    from_domain_sid: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class ByocTrunkInstance {
    protected _version: V1;
    protected _solution: ByocTrunkContextSolution;
    protected _context?: ByocTrunkContext;
    constructor(_version: V1, payload: ByocTrunkResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the BYOC Trunk resource.
     */
    accountSid: string;
    /**
     * The unique string that that we created to identify the BYOC Trunk resource.
     */
    sid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The URL we call using the `voice_method` when the BYOC Trunk receives a call.
     */
    voiceUrl: string;
    /**
     * The HTTP method we use to call `voice_url`. Can be: `GET` or `POST`.
     */
    voiceMethod: string;
    /**
     * The URL that we call when an error occurs while retrieving or executing the TwiML requested from `voice_url`.
     */
    voiceFallbackUrl: string;
    /**
     * The HTTP method we use to call `voice_fallback_url`. Can be: `GET` or `POST`.
     */
    voiceFallbackMethod: string;
    /**
     * The URL that we call to pass status parameters (such as call ended) to your application.
     */
    statusCallbackUrl: string;
    /**
     * The HTTP method we use to call `status_callback_url`. Either `GET` or `POST`.
     */
    statusCallbackMethod: string;
    /**
     * Whether Caller ID Name (CNAM) lookup is enabled for the trunk. If enabled, all inbound calls to the BYOC Trunk from the United States and Canada automatically perform a CNAM Lookup and display Caller ID data on your phone. See [CNAM Lookups](https://www.twilio.com/docs/sip-trunking#CNAM) for more information.
     */
    cnamLookupEnabled: boolean;
    /**
     * The SID of the Connection Policy that Twilio will use when routing traffic to your communications infrastructure.
     */
    connectionPolicySid: string;
    /**
     * The SID of the SIP Domain that should be used in the `From` header of originating calls sent to your SIP infrastructure. If your SIP infrastructure allows users to \"call back\" an incoming call, configure this with a [SIP Domain](https://www.twilio.com/docs/voice/api/sending-sip) to ensure proper routing. If not configured, the from domain will default to \"sip.twilio.com\".
     */
    fromDomainSid: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a ByocTrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ByocTrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ByocTrunkInstance
     */
    fetch(callback?: (error: Error | null, item?: ByocTrunkInstance) => any): Promise<ByocTrunkInstance>;
    /**
     * Update a ByocTrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ByocTrunkInstance
     */
    update(callback?: (error: Error | null, item?: ByocTrunkInstance) => any): Promise<ByocTrunkInstance>;
    /**
     * Update a ByocTrunkInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ByocTrunkInstance
     */
    update(params: ByocTrunkContextUpdateOptions, callback?: (error: Error | null, item?: ByocTrunkInstance) => any): Promise<ByocTrunkInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        sid: string;
        friendlyName: string;
        voiceUrl: string;
        voiceMethod: string;
        voiceFallbackUrl: string;
        voiceFallbackMethod: string;
        statusCallbackUrl: string;
        statusCallbackMethod: string;
        cnamLookupEnabled: boolean;
        connectionPolicySid: string;
        fromDomainSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ByocTrunkSolution {
}
export interface ByocTrunkListInstance {
    _version: V1;
    _solution: ByocTrunkSolution;
    _uri: string;
    (sid: string): ByocTrunkContext;
    get(sid: string): ByocTrunkContext;
    /**
     * Create a ByocTrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ByocTrunkInstance
     */
    create(callback?: (error: Error | null, item?: ByocTrunkInstance) => any): Promise<ByocTrunkInstance>;
    /**
     * Create a ByocTrunkInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ByocTrunkInstance
     */
    create(params: ByocTrunkListInstanceCreateOptions, callback?: (error: Error | null, item?: ByocTrunkInstance) => any): Promise<ByocTrunkInstance>;
    /**
     * Streams ByocTrunkInstance records from the API.
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
     * @param { ByocTrunkListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ByocTrunkInstance, done: (err?: Error) => void) => void): void;
    each(params: ByocTrunkListInstanceEachOptions, callback?: (item: ByocTrunkInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ByocTrunkInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ByocTrunkPage) => any): Promise<ByocTrunkPage>;
    /**
     * Lists ByocTrunkInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ByocTrunkListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ByocTrunkInstance[]) => any): Promise<ByocTrunkInstance[]>;
    list(params: ByocTrunkListInstanceOptions, callback?: (error: Error | null, items: ByocTrunkInstance[]) => any): Promise<ByocTrunkInstance[]>;
    /**
     * Retrieve a single page of ByocTrunkInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ByocTrunkListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ByocTrunkPage) => any): Promise<ByocTrunkPage>;
    page(params: ByocTrunkListInstancePageOptions, callback?: (error: Error | null, items: ByocTrunkPage) => any): Promise<ByocTrunkPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ByocTrunkListInstance(version: V1): ByocTrunkListInstance;
export declare class ByocTrunkPage extends Page<V1, ByocTrunkPayload, ByocTrunkResource, ByocTrunkInstance> {
    /**
     * Initialize the ByocTrunkPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ByocTrunkSolution);
    /**
     * Build an instance of ByocTrunkInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ByocTrunkResource): ByocTrunkInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
