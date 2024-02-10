/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { CredentialListListInstance } from "./trunk/credentialList";
import { IpAccessControlListListInstance } from "./trunk/ipAccessControlList";
import { OriginationUrlListInstance } from "./trunk/originationUrl";
import { PhoneNumberListInstance } from "./trunk/phoneNumber";
import { RecordingListInstance } from "./trunk/recording";
export type TrunkTransferCallerId = "from-transferee" | "from-transferor";
export type TrunkTransferSetting = "disable-all" | "enable-all" | "sip-only";
/**
 * Options to pass to update a TrunkInstance
 */
export interface TrunkContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** The unique address you reserve on Twilio to which you route your SIP traffic. Domain names can contain letters, digits, and `-` and must end with `pstn.twilio.com`. See [Termination Settings](https://www.twilio.com/docs/sip-trunking#termination) for more information. */
    domainName?: string;
    /** The URL we should call using the `disaster_recovery_method` if an error occurs while sending SIP traffic towards the configured Origination URL. We retrieve TwiML from the URL and execute the instructions like any other normal TwiML call. See [Disaster Recovery](https://www.twilio.com/docs/sip-trunking#disaster-recovery) for more information. */
    disasterRecoveryUrl?: string;
    /** The HTTP method we should use to call the `disaster_recovery_url`. Can be: `GET` or `POST`. */
    disasterRecoveryMethod?: string;
    /**  */
    transferMode?: TrunkTransferSetting;
    /** Whether Secure Trunking is enabled for the trunk. If enabled, all calls going through the trunk will be secure using SRTP for media and TLS for signaling. If disabled, then RTP will be used for media. See [Secure Trunking](https://www.twilio.com/docs/sip-trunking#securetrunking) for more information. */
    secure?: boolean;
    /** Whether Caller ID Name (CNAM) lookup should be enabled for the trunk. If enabled, all inbound calls to the SIP Trunk from the United States and Canada automatically perform a CNAM Lookup and display Caller ID data on your phone. See [CNAM Lookups](https://www.twilio.com/docs/sip-trunking#CNAM) for more information. */
    cnamLookupEnabled?: boolean;
    /**  */
    transferCallerId?: TrunkTransferCallerId;
}
/**
 * Options to pass to create a TrunkInstance
 */
export interface TrunkListInstanceCreateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** The unique address you reserve on Twilio to which you route your SIP traffic. Domain names can contain letters, digits, and `-` and must end with `pstn.twilio.com`. See [Termination Settings](https://www.twilio.com/docs/sip-trunking#termination) for more information. */
    domainName?: string;
    /** The URL we should call using the `disaster_recovery_method` if an error occurs while sending SIP traffic towards the configured Origination URL. We retrieve TwiML from the URL and execute the instructions like any other normal TwiML call. See [Disaster Recovery](https://www.twilio.com/docs/sip-trunking#disaster-recovery) for more information. */
    disasterRecoveryUrl?: string;
    /** The HTTP method we should use to call the `disaster_recovery_url`. Can be: `GET` or `POST`. */
    disasterRecoveryMethod?: string;
    /**  */
    transferMode?: TrunkTransferSetting;
    /** Whether Secure Trunking is enabled for the trunk. If enabled, all calls going through the trunk will be secure using SRTP for media and TLS for signaling. If disabled, then RTP will be used for media. See [Secure Trunking](https://www.twilio.com/docs/sip-trunking#securetrunking) for more information. */
    secure?: boolean;
    /** Whether Caller ID Name (CNAM) lookup should be enabled for the trunk. If enabled, all inbound calls to the SIP Trunk from the United States and Canada automatically perform a CNAM Lookup and display Caller ID data on your phone. See [CNAM Lookups](https://www.twilio.com/docs/sip-trunking#CNAM) for more information. */
    cnamLookupEnabled?: boolean;
    /**  */
    transferCallerId?: TrunkTransferCallerId;
}
/**
 * Options to pass to each
 */
export interface TrunkListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TrunkInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TrunkListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TrunkListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TrunkContext {
    credentialsLists: CredentialListListInstance;
    ipAccessControlLists: IpAccessControlListListInstance;
    originationUrls: OriginationUrlListInstance;
    phoneNumbers: PhoneNumberListInstance;
    recordings: RecordingListInstance;
    /**
     * Remove a TrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    fetch(callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Update a TrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    update(callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Update a TrunkInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    update(params: TrunkContextUpdateOptions, callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TrunkContextSolution {
    sid: string;
}
export declare class TrunkContextImpl implements TrunkContext {
    protected _version: V1;
    protected _solution: TrunkContextSolution;
    protected _uri: string;
    protected _credentialsLists?: CredentialListListInstance;
    protected _ipAccessControlLists?: IpAccessControlListListInstance;
    protected _originationUrls?: OriginationUrlListInstance;
    protected _phoneNumbers?: PhoneNumberListInstance;
    protected _recordings?: RecordingListInstance;
    constructor(_version: V1, sid: string);
    get credentialsLists(): CredentialListListInstance;
    get ipAccessControlLists(): IpAccessControlListListInstance;
    get originationUrls(): OriginationUrlListInstance;
    get phoneNumbers(): PhoneNumberListInstance;
    get recordings(): RecordingListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    update(params?: TrunkContextUpdateOptions | ((error: Error | null, item?: TrunkInstance) => any), callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TrunkContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TrunkPayload extends TwilioResponsePayload {
    trunks: TrunkResource[];
}
interface TrunkResource {
    account_sid: string;
    domain_name: string;
    disaster_recovery_method: string;
    disaster_recovery_url: string;
    friendly_name: string;
    secure: boolean;
    recording: any;
    transfer_mode: TrunkTransferSetting;
    transfer_caller_id: TrunkTransferCallerId;
    cnam_lookup_enabled: boolean;
    auth_type: string;
    auth_type_set: Array<string>;
    date_created: Date;
    date_updated: Date;
    sid: string;
    url: string;
    links: Record<string, string>;
}
export declare class TrunkInstance {
    protected _version: V1;
    protected _solution: TrunkContextSolution;
    protected _context?: TrunkContext;
    constructor(_version: V1, payload: TrunkResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Trunk resource.
     */
    accountSid: string;
    /**
     * The unique address you reserve on Twilio to which you route your SIP traffic. Domain names can contain letters, digits, and `-` and must end with `pstn.twilio.com`. See [Termination Settings](https://www.twilio.com/docs/sip-trunking#termination) for more information.
     */
    domainName: string;
    /**
     * The HTTP method we use to call the `disaster_recovery_url`. Can be: `GET` or `POST`.
     */
    disasterRecoveryMethod: string;
    /**
     * The URL we call using the `disaster_recovery_method` if an error occurs while sending SIP traffic towards the configured Origination URL. We retrieve TwiML from this URL and execute the instructions like any other normal TwiML call. See [Disaster Recovery](https://www.twilio.com/docs/sip-trunking#disaster-recovery) for more information.
     */
    disasterRecoveryUrl: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * Whether Secure Trunking is enabled for the trunk. If enabled, all calls going through the trunk will be secure using SRTP for media and TLS for signaling. If disabled, then RTP will be used for media. See [Secure Trunking](https://www.twilio.com/docs/sip-trunking#securetrunking) for more information.
     */
    secure: boolean;
    /**
     * The recording settings for the trunk. Can be: `do-not-record`, `record-from-ringing`, `record-from-answer`. If set to `record-from-ringing` or `record-from-answer`, all calls going through the trunk will be recorded. The only way to change recording parameters is on a sub-resource of a Trunk after it has been created. e.g.`/Trunks/[Trunk_SID]/Recording -XPOST -d\'Mode=record-from-answer\'`. See [Recording](https://www.twilio.com/docs/sip-trunking#recording) for more information.
     */
    recording: any;
    transferMode: TrunkTransferSetting;
    transferCallerId: TrunkTransferCallerId;
    /**
     * Whether Caller ID Name (CNAM) lookup is enabled for the trunk. If enabled, all inbound calls to the SIP Trunk from the United States and Canada automatically perform a CNAM Lookup and display Caller ID data on your phone. See [CNAM Lookups](https://www.twilio.com/docs/sip-trunking#CNAM) for more information.
     */
    cnamLookupEnabled: boolean;
    /**
     * The types of authentication mapped to the domain. Can be: `IP_ACL` and `CREDENTIAL_LIST`. If both are mapped, the values are returned in a comma delimited list. If empty, the domain will not receive any traffic.
     */
    authType: string;
    /**
     * Reserved.
     */
    authTypeSet: Array<string>;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The unique string that we created to identify the Trunk resource.
     */
    sid: string;
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
     * Remove a TrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    fetch(callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Update a TrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    update(callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Update a TrunkInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    update(params: TrunkContextUpdateOptions, callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Access the credentialsLists.
     */
    credentialsLists(): CredentialListListInstance;
    /**
     * Access the ipAccessControlLists.
     */
    ipAccessControlLists(): IpAccessControlListListInstance;
    /**
     * Access the originationUrls.
     */
    originationUrls(): OriginationUrlListInstance;
    /**
     * Access the phoneNumbers.
     */
    phoneNumbers(): PhoneNumberListInstance;
    /**
     * Access the recordings.
     */
    recordings(): RecordingListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        domainName: string;
        disasterRecoveryMethod: string;
        disasterRecoveryUrl: string;
        friendlyName: string;
        secure: boolean;
        recording: any;
        transferMode: TrunkTransferSetting;
        transferCallerId: TrunkTransferCallerId;
        cnamLookupEnabled: boolean;
        authType: string;
        authTypeSet: string[];
        dateCreated: Date;
        dateUpdated: Date;
        sid: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TrunkSolution {
}
export interface TrunkListInstance {
    _version: V1;
    _solution: TrunkSolution;
    _uri: string;
    (sid: string): TrunkContext;
    get(sid: string): TrunkContext;
    /**
     * Create a TrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    create(callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Create a TrunkInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    create(params: TrunkListInstanceCreateOptions, callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Streams TrunkInstance records from the API.
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
     * @param { TrunkListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TrunkInstance, done: (err?: Error) => void) => void): void;
    each(params: TrunkListInstanceEachOptions, callback?: (item: TrunkInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TrunkInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TrunkPage) => any): Promise<TrunkPage>;
    /**
     * Lists TrunkInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TrunkListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TrunkInstance[]) => any): Promise<TrunkInstance[]>;
    list(params: TrunkListInstanceOptions, callback?: (error: Error | null, items: TrunkInstance[]) => any): Promise<TrunkInstance[]>;
    /**
     * Retrieve a single page of TrunkInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TrunkListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TrunkPage) => any): Promise<TrunkPage>;
    page(params: TrunkListInstancePageOptions, callback?: (error: Error | null, items: TrunkPage) => any): Promise<TrunkPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TrunkListInstance(version: V1): TrunkListInstance;
export declare class TrunkPage extends Page<V1, TrunkPayload, TrunkResource, TrunkInstance> {
    /**
     * Initialize the TrunkPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: TrunkSolution);
    /**
     * Build an instance of TrunkInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TrunkResource): TrunkInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
