/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2 from "../../../V2";
import { NotificationListInstance } from "./challenge/notification";
export type ChallengeChallengeReasons = "none" | "not_needed" | "not_requested";
export type ChallengeChallengeStatuses = "pending" | "expired" | "approved" | "denied";
export type ChallengeFactorTypes = "push" | "totp";
export type ChallengeListOrders = "asc" | "desc";
/**
 * Options to pass to update a ChallengeInstance
 */
export interface ChallengeContextUpdateOptions {
    /** The optional payload needed to verify the Challenge. E.g., a TOTP would use the numeric code. For `TOTP` this value must be between 3 and 8 characters long. For `Push` this value can be up to 5456 characters in length */
    authPayload?: string;
    /** Custom metadata associated with the challenge. This is added by the Device/SDK directly to allow for the inclusion of device information. It must be a stringified JSON with only strings values eg. `{\\\"os\\\": \\\"Android\\\"}`. Can be up to 1024 characters in length. */
    metadata?: any;
}
/**
 * Options to pass to create a ChallengeInstance
 */
export interface ChallengeListInstanceCreateOptions {
    /** The unique SID identifier of the Factor. */
    factorSid: string;
    /** The date-time when this Challenge expires, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. The default value is five (5) minutes after Challenge creation. The max value is sixty (60) minutes after creation. */
    expirationDate?: Date;
    /** Shown to the user when the push notification arrives. Required when `factor_type` is `push`. Can be up to 256 characters in length */
    "details.message"?: string;
    /** A list of objects that describe the Fields included in the Challenge. Each object contains the label and value of the field, the label can be up to 36 characters in length and the value can be up to 128 characters in length. Used when `factor_type` is `push`. There can be up to 20 details fields. */
    "details.fields"?: Array<any>;
    /** Details provided to give context about the Challenge. Not shown to the end user. It must be a stringified JSON with only strings values eg. `{\\\"ip\\\": \\\"172.168.1.234\\\"}`. Can be up to 1024 characters in length */
    hiddenDetails?: any;
    /** Optional payload used to verify the Challenge upon creation. Only used with a Factor of type `totp` to carry the TOTP code that needs to be verified. For `TOTP` this value must be between 3 and 8 characters long. */
    authPayload?: string;
}
/**
 * Options to pass to each
 */
export interface ChallengeListInstanceEachOptions {
    /** The unique SID identifier of the Factor. */
    factorSid?: string;
    /** The Status of the Challenges to fetch. One of `pending`, `expired`, `approved` or `denied`. */
    status?: ChallengeChallengeStatuses;
    /** The desired sort order of the Challenges list. One of `asc` or `desc` for ascending and descending respectively. Defaults to `asc`. */
    order?: ChallengeListOrders;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ChallengeInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ChallengeListInstanceOptions {
    /** The unique SID identifier of the Factor. */
    factorSid?: string;
    /** The Status of the Challenges to fetch. One of `pending`, `expired`, `approved` or `denied`. */
    status?: ChallengeChallengeStatuses;
    /** The desired sort order of the Challenges list. One of `asc` or `desc` for ascending and descending respectively. Defaults to `asc`. */
    order?: ChallengeListOrders;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ChallengeListInstancePageOptions {
    /** The unique SID identifier of the Factor. */
    factorSid?: string;
    /** The Status of the Challenges to fetch. One of `pending`, `expired`, `approved` or `denied`. */
    status?: ChallengeChallengeStatuses;
    /** The desired sort order of the Challenges list. One of `asc` or `desc` for ascending and descending respectively. Defaults to `asc`. */
    order?: ChallengeListOrders;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ChallengeContext {
    notifications: NotificationListInstance;
    /**
     * Fetch a ChallengeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChallengeInstance
     */
    fetch(callback?: (error: Error | null, item?: ChallengeInstance) => any): Promise<ChallengeInstance>;
    /**
     * Update a ChallengeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChallengeInstance
     */
    update(callback?: (error: Error | null, item?: ChallengeInstance) => any): Promise<ChallengeInstance>;
    /**
     * Update a ChallengeInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChallengeInstance
     */
    update(params: ChallengeContextUpdateOptions, callback?: (error: Error | null, item?: ChallengeInstance) => any): Promise<ChallengeInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ChallengeContextSolution {
    serviceSid: string;
    identity: string;
    sid: string;
}
export declare class ChallengeContextImpl implements ChallengeContext {
    protected _version: V2;
    protected _solution: ChallengeContextSolution;
    protected _uri: string;
    protected _notifications?: NotificationListInstance;
    constructor(_version: V2, serviceSid: string, identity: string, sid: string);
    get notifications(): NotificationListInstance;
    fetch(callback?: (error: Error | null, item?: ChallengeInstance) => any): Promise<ChallengeInstance>;
    update(params?: ChallengeContextUpdateOptions | ((error: Error | null, item?: ChallengeInstance) => any), callback?: (error: Error | null, item?: ChallengeInstance) => any): Promise<ChallengeInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ChallengeContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ChallengePayload extends TwilioResponsePayload {
    challenges: ChallengeResource[];
}
interface ChallengeResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    entity_sid: string;
    identity: string;
    factor_sid: string;
    date_created: Date;
    date_updated: Date;
    date_responded: Date;
    expiration_date: Date;
    status: ChallengeChallengeStatuses;
    responded_reason: ChallengeChallengeReasons;
    details: any;
    hidden_details: any;
    metadata: any;
    factor_type: ChallengeFactorTypes;
    url: string;
    links: Record<string, string>;
}
export declare class ChallengeInstance {
    protected _version: V2;
    protected _solution: ChallengeContextSolution;
    protected _context?: ChallengeContext;
    constructor(_version: V2, payload: ChallengeResource, serviceSid: string, identity: string, sid?: string);
    /**
     * A 34 character string that uniquely identifies this Challenge.
     */
    sid: string;
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * The unique SID identifier of the Service.
     */
    serviceSid: string;
    /**
     * The unique SID identifier of the Entity.
     */
    entitySid: string;
    /**
     * Customer unique identity for the Entity owner of the Challenge. This identifier should be immutable, not PII, length between 8 and 64 characters, and generated by your external system, such as your user\'s UUID, GUID, or SID. It can only contain dash (-) separated alphanumeric characters.
     */
    identity: string;
    /**
     * The unique SID identifier of the Factor.
     */
    factorSid: string;
    /**
     * The date that this Challenge was created, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date that this Challenge was updated, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The date that this Challenge was responded, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateResponded: Date;
    /**
     * The date-time when this Challenge expires, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. The default value is five (5) minutes after Challenge creation. The max value is sixty (60) minutes after creation.
     */
    expirationDate: Date;
    status: ChallengeChallengeStatuses;
    respondedReason: ChallengeChallengeReasons;
    /**
     * Details provided to give context about the Challenge. Intended to be shown to the end user.
     */
    details: any;
    /**
     * Details provided to give context about the Challenge. Intended to be hidden from the end user. It must be a stringified JSON with only strings values eg. `{\"ip\": \"172.168.1.234\"}`
     */
    hiddenDetails: any;
    /**
     * Custom metadata associated with the challenge. This is added by the Device/SDK directly to allow for the inclusion of device information. It must be a stringified JSON with only strings values eg. `{\"os\": \"Android\"}`. Can be up to 1024 characters in length.
     */
    metadata: any;
    factorType: ChallengeFactorTypes;
    /**
     * The URL of this resource.
     */
    url: string;
    /**
     * Contains a dictionary of URL links to nested resources of this Challenge.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a ChallengeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChallengeInstance
     */
    fetch(callback?: (error: Error | null, item?: ChallengeInstance) => any): Promise<ChallengeInstance>;
    /**
     * Update a ChallengeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChallengeInstance
     */
    update(callback?: (error: Error | null, item?: ChallengeInstance) => any): Promise<ChallengeInstance>;
    /**
     * Update a ChallengeInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChallengeInstance
     */
    update(params: ChallengeContextUpdateOptions, callback?: (error: Error | null, item?: ChallengeInstance) => any): Promise<ChallengeInstance>;
    /**
     * Access the notifications.
     */
    notifications(): NotificationListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        entitySid: string;
        identity: string;
        factorSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        dateResponded: Date;
        expirationDate: Date;
        status: ChallengeChallengeStatuses;
        respondedReason: ChallengeChallengeReasons;
        details: any;
        hiddenDetails: any;
        metadata: any;
        factorType: ChallengeFactorTypes;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ChallengeSolution {
    serviceSid: string;
    identity: string;
}
export interface ChallengeListInstance {
    _version: V2;
    _solution: ChallengeSolution;
    _uri: string;
    (sid: string): ChallengeContext;
    get(sid: string): ChallengeContext;
    /**
     * Create a ChallengeInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChallengeInstance
     */
    create(params: ChallengeListInstanceCreateOptions, callback?: (error: Error | null, item?: ChallengeInstance) => any): Promise<ChallengeInstance>;
    /**
     * Streams ChallengeInstance records from the API.
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
     * @param { ChallengeListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ChallengeInstance, done: (err?: Error) => void) => void): void;
    each(params: ChallengeListInstanceEachOptions, callback?: (item: ChallengeInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ChallengeInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ChallengePage) => any): Promise<ChallengePage>;
    /**
     * Lists ChallengeInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ChallengeListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ChallengeInstance[]) => any): Promise<ChallengeInstance[]>;
    list(params: ChallengeListInstanceOptions, callback?: (error: Error | null, items: ChallengeInstance[]) => any): Promise<ChallengeInstance[]>;
    /**
     * Retrieve a single page of ChallengeInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ChallengeListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ChallengePage) => any): Promise<ChallengePage>;
    page(params: ChallengeListInstancePageOptions, callback?: (error: Error | null, items: ChallengePage) => any): Promise<ChallengePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ChallengeListInstance(version: V2, serviceSid: string, identity: string): ChallengeListInstance;
export declare class ChallengePage extends Page<V2, ChallengePayload, ChallengeResource, ChallengeInstance> {
    /**
     * Initialize the ChallengePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: ChallengeSolution);
    /**
     * Build an instance of ChallengeInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ChallengeResource): ChallengeInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
