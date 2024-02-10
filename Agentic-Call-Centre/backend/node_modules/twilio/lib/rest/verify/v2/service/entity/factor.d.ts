/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2 from "../../../V2";
export type FactorFactorStatuses = "unverified" | "verified";
export type FactorFactorTypes = "push" | "totp";
export type FactorTotpAlgorithms = "sha1" | "sha256" | "sha512";
/**
 * Options to pass to update a FactorInstance
 */
export interface FactorContextUpdateOptions {
    /** The optional payload needed to verify the Factor for the first time. E.g. for a TOTP, the numeric code. */
    authPayload?: string;
    /** The new friendly name of this Factor. It can be up to 64 characters. */
    friendlyName?: string;
    /** For APN, the device token. For FCM, the registration token. It is used to send the push notifications. Required when `factor_type` is `push`. If specified, this value must be between 32 and 255 characters long. */
    "config.notificationToken"?: string;
    /** The Verify Push SDK version used to configure the factor */
    "config.sdkVersion"?: string;
    /** Defines how often, in seconds, are TOTP codes generated. i.e, a new TOTP code is generated every time_step seconds. Must be between 20 and 60 seconds, inclusive */
    "config.timeStep"?: number;
    /** The number of time-steps, past and future, that are valid for validation of TOTP codes. Must be between 0 and 2, inclusive */
    "config.skew"?: number;
    /** Number of digits for generated TOTP codes. Must be between 3 and 8, inclusive */
    "config.codeLength"?: number;
    /**  */
    "config.alg"?: FactorTotpAlgorithms;
    /** The transport technology used to generate the Notification Token. Can be `apn`, `fcm` or `none`.  Required when `factor_type` is `push`. */
    "config.notificationPlatform"?: string;
}
/**
 * Options to pass to each
 */
export interface FactorListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: FactorInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface FactorListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface FactorListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface FactorContext {
    /**
     * Remove a FactorInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a FactorInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FactorInstance
     */
    fetch(callback?: (error: Error | null, item?: FactorInstance) => any): Promise<FactorInstance>;
    /**
     * Update a FactorInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FactorInstance
     */
    update(callback?: (error: Error | null, item?: FactorInstance) => any): Promise<FactorInstance>;
    /**
     * Update a FactorInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FactorInstance
     */
    update(params: FactorContextUpdateOptions, callback?: (error: Error | null, item?: FactorInstance) => any): Promise<FactorInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface FactorContextSolution {
    serviceSid: string;
    identity: string;
    sid: string;
}
export declare class FactorContextImpl implements FactorContext {
    protected _version: V2;
    protected _solution: FactorContextSolution;
    protected _uri: string;
    constructor(_version: V2, serviceSid: string, identity: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: FactorInstance) => any): Promise<FactorInstance>;
    update(params?: FactorContextUpdateOptions | ((error: Error | null, item?: FactorInstance) => any), callback?: (error: Error | null, item?: FactorInstance) => any): Promise<FactorInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): FactorContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface FactorPayload extends TwilioResponsePayload {
    factors: FactorResource[];
}
interface FactorResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    entity_sid: string;
    identity: string;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    status: FactorFactorStatuses;
    factor_type: FactorFactorTypes;
    config: any;
    metadata: any;
    url: string;
}
export declare class FactorInstance {
    protected _version: V2;
    protected _solution: FactorContextSolution;
    protected _context?: FactorContext;
    constructor(_version: V2, payload: FactorResource, serviceSid: string, identity: string, sid?: string);
    /**
     * A 34 character string that uniquely identifies this Factor.
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
     * Customer unique identity for the Entity owner of the Factor. This identifier should be immutable, not PII, length between 8 and 64 characters, and generated by your external system, such as your user\'s UUID, GUID, or SID. It can only contain dash (-) separated alphanumeric characters.
     */
    identity: string;
    /**
     * The date that this Factor was created, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date that this Factor was updated, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * A human readable description of this resource, up to 64 characters. For a push factor, this can be the device\'s name.
     */
    friendlyName: string;
    status: FactorFactorStatuses;
    factorType: FactorFactorTypes;
    /**
     * An object that contains configurations specific to a `factor_type`.
     */
    config: any;
    /**
     * Custom metadata associated with the factor. This is added by the Device/SDK directly to allow for the inclusion of device information. It must be a stringified JSON with only strings values eg. `{\"os\": \"Android\"}`. Can be up to 1024 characters in length.
     */
    metadata: any;
    /**
     * The URL of this resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a FactorInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a FactorInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FactorInstance
     */
    fetch(callback?: (error: Error | null, item?: FactorInstance) => any): Promise<FactorInstance>;
    /**
     * Update a FactorInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FactorInstance
     */
    update(callback?: (error: Error | null, item?: FactorInstance) => any): Promise<FactorInstance>;
    /**
     * Update a FactorInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FactorInstance
     */
    update(params: FactorContextUpdateOptions, callback?: (error: Error | null, item?: FactorInstance) => any): Promise<FactorInstance>;
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
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        status: FactorFactorStatuses;
        factorType: FactorFactorTypes;
        config: any;
        metadata: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface FactorSolution {
    serviceSid: string;
    identity: string;
}
export interface FactorListInstance {
    _version: V2;
    _solution: FactorSolution;
    _uri: string;
    (sid: string): FactorContext;
    get(sid: string): FactorContext;
    /**
     * Streams FactorInstance records from the API.
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
     * @param { FactorListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: FactorInstance, done: (err?: Error) => void) => void): void;
    each(params: FactorListInstanceEachOptions, callback?: (item: FactorInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of FactorInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: FactorPage) => any): Promise<FactorPage>;
    /**
     * Lists FactorInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FactorListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: FactorInstance[]) => any): Promise<FactorInstance[]>;
    list(params: FactorListInstanceOptions, callback?: (error: Error | null, items: FactorInstance[]) => any): Promise<FactorInstance[]>;
    /**
     * Retrieve a single page of FactorInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FactorListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: FactorPage) => any): Promise<FactorPage>;
    page(params: FactorListInstancePageOptions, callback?: (error: Error | null, items: FactorPage) => any): Promise<FactorPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function FactorListInstance(version: V2, serviceSid: string, identity: string): FactorListInstance;
export declare class FactorPage extends Page<V2, FactorPayload, FactorResource, FactorInstance> {
    /**
     * Initialize the FactorPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: FactorSolution);
    /**
     * Build an instance of FactorInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: FactorResource): FactorInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
