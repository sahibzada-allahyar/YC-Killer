/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { DataSessionListInstance } from "./sim/dataSession";
import { UsageRecordListInstance } from "./sim/usageRecord";
export type SimResetStatus = "resetting";
export type SimStatus = "new" | "ready" | "active" | "suspended" | "deactivated" | "canceled" | "scheduled" | "updating";
/**
 * Options to pass to update a SimInstance
 */
export interface SimContextUpdateOptions {
    /** An application-defined string that uniquely identifies the resource. It can be used in place of the `sid` in the URL path to address the resource. */
    uniqueName?: string;
    /** The HTTP method we should use to call `callback_url`. Can be: `POST` or `GET`. The default is `POST`. */
    callbackMethod?: string;
    /** The URL we should call using the `callback_url` when the SIM has finished updating. When the SIM transitions from `new` to `ready` or from any status to `deactivated`, we call this URL when the status changes to an intermediate status (`ready` or `deactivated`) and again when the status changes to its final status (`active` or `canceled`). */
    callbackUrl?: string;
    /** A descriptive string that you create to describe the Sim resource. It does not need to be unique. */
    friendlyName?: string;
    /** The SID or unique name of the [RatePlan resource](https://www.twilio.com/docs/iot/wireless/api/rateplan-resource) to which the Sim resource should be assigned. */
    ratePlan?: string;
    /**  */
    status?: SimStatus;
    /** The HTTP method we should use to call `commands_callback_url`. Can be: `POST` or `GET`. The default is `POST`. */
    commandsCallbackMethod?: string;
    /** The URL we should call using the `commands_callback_method` when the SIM sends a [Command](https://www.twilio.com/docs/iot/wireless/api/command-resource). Your server should respond with an HTTP status code in the 200 range; any response body is ignored. */
    commandsCallbackUrl?: string;
    /** The HTTP method we should use to call `sms_fallback_url`. Can be: `GET` or `POST`. Default is `POST`. */
    smsFallbackMethod?: string;
    /** The URL we should call using the `sms_fallback_method` when an error occurs while retrieving or executing the TwiML requested from `sms_url`. */
    smsFallbackUrl?: string;
    /** The HTTP method we should use to call `sms_url`. Can be: `GET` or `POST`. Default is `POST`. */
    smsMethod?: string;
    /** The URL we should call using the `sms_method` when the SIM-connected device sends an SMS message that is not a [Command](https://www.twilio.com/docs/iot/wireless/api/command-resource). */
    smsUrl?: string;
    /** Deprecated. */
    voiceFallbackMethod?: string;
    /** Deprecated. */
    voiceFallbackUrl?: string;
    /** Deprecated. */
    voiceMethod?: string;
    /** Deprecated. */
    voiceUrl?: string;
    /**  */
    resetStatus?: SimResetStatus;
    /** The SID of the [Account](https://www.twilio.com/docs/iam/api/account) to which the Sim resource should belong. The Account SID can only be that of the requesting Account or that of a [Subaccount](https://www.twilio.com/docs/iam/api/subaccounts) of the requesting Account. Only valid when the Sim resource\\\'s status is `new`. For more information, see the [Move SIMs between Subaccounts documentation](https://www.twilio.com/docs/iot/wireless/api/sim-resource#move-sims-between-subaccounts). */
    accountSid?: string;
}
/**
 * Options to pass to each
 */
export interface SimListInstanceEachOptions {
    /** Only return Sim resources with this status. */
    status?: SimStatus;
    /** Only return Sim resources with this ICCID. This will return a list with a maximum size of 1. */
    iccid?: string;
    /** The SID or unique name of a [RatePlan resource](https://www.twilio.com/docs/iot/wireless/api/rateplan-resource). Only return Sim resources assigned to this RatePlan resource. */
    ratePlan?: string;
    /** Deprecated. */
    eId?: string;
    /** Only return Sim resources with this registration code. This will return a list with a maximum size of 1. */
    simRegistrationCode?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SimInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SimListInstanceOptions {
    /** Only return Sim resources with this status. */
    status?: SimStatus;
    /** Only return Sim resources with this ICCID. This will return a list with a maximum size of 1. */
    iccid?: string;
    /** The SID or unique name of a [RatePlan resource](https://www.twilio.com/docs/iot/wireless/api/rateplan-resource). Only return Sim resources assigned to this RatePlan resource. */
    ratePlan?: string;
    /** Deprecated. */
    eId?: string;
    /** Only return Sim resources with this registration code. This will return a list with a maximum size of 1. */
    simRegistrationCode?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SimListInstancePageOptions {
    /** Only return Sim resources with this status. */
    status?: SimStatus;
    /** Only return Sim resources with this ICCID. This will return a list with a maximum size of 1. */
    iccid?: string;
    /** The SID or unique name of a [RatePlan resource](https://www.twilio.com/docs/iot/wireless/api/rateplan-resource). Only return Sim resources assigned to this RatePlan resource. */
    ratePlan?: string;
    /** Deprecated. */
    eId?: string;
    /** Only return Sim resources with this registration code. This will return a list with a maximum size of 1. */
    simRegistrationCode?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SimContext {
    dataSessions: DataSessionListInstance;
    usageRecords: UsageRecordListInstance;
    /**
     * Remove a SimInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SimInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SimInstance
     */
    fetch(callback?: (error: Error | null, item?: SimInstance) => any): Promise<SimInstance>;
    /**
     * Update a SimInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SimInstance
     */
    update(callback?: (error: Error | null, item?: SimInstance) => any): Promise<SimInstance>;
    /**
     * Update a SimInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SimInstance
     */
    update(params: SimContextUpdateOptions, callback?: (error: Error | null, item?: SimInstance) => any): Promise<SimInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SimContextSolution {
    sid: string;
}
export declare class SimContextImpl implements SimContext {
    protected _version: V1;
    protected _solution: SimContextSolution;
    protected _uri: string;
    protected _dataSessions?: DataSessionListInstance;
    protected _usageRecords?: UsageRecordListInstance;
    constructor(_version: V1, sid: string);
    get dataSessions(): DataSessionListInstance;
    get usageRecords(): UsageRecordListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SimInstance) => any): Promise<SimInstance>;
    update(params?: SimContextUpdateOptions | ((error: Error | null, item?: SimInstance) => any), callback?: (error: Error | null, item?: SimInstance) => any): Promise<SimInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SimContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SimPayload extends TwilioResponsePayload {
    sims: SimResource[];
}
interface SimResource {
    sid: string;
    unique_name: string;
    account_sid: string;
    rate_plan_sid: string;
    friendly_name: string;
    iccid: string;
    e_id: string;
    status: SimStatus;
    reset_status: SimResetStatus;
    commands_callback_url: string;
    commands_callback_method: string;
    sms_fallback_method: string;
    sms_fallback_url: string;
    sms_method: string;
    sms_url: string;
    voice_fallback_method: string;
    voice_fallback_url: string;
    voice_method: string;
    voice_url: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
    ip_address: string;
}
export declare class SimInstance {
    protected _version: V1;
    protected _solution: SimContextSolution;
    protected _context?: SimContext;
    constructor(_version: V1, payload: SimResource, sid?: string);
    /**
     * The unique string that we created to identify the Sim resource.
     */
    sid: string;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used in place of the resource\'s `sid` in the URL to address the resource.
     */
    uniqueName: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) to which the Sim resource belongs.
     */
    accountSid: string;
    /**
     * The SID of the [RatePlan resource](https://www.twilio.com/docs/iot/wireless/api/rateplan-resource) to which the Sim resource is assigned.
     */
    ratePlanSid: string;
    /**
     * The string that you assigned to describe the Sim resource.
     */
    friendlyName: string;
    /**
     * The [ICCID](https://en.wikipedia.org/wiki/SIM_card#ICCID) associated with the SIM.
     */
    iccid: string;
    /**
     * Deprecated.
     */
    eId: string;
    status: SimStatus;
    resetStatus: SimResetStatus;
    /**
     * The URL we call using the `commands_callback_method` when the SIM originates a machine-to-machine [Command](https://www.twilio.com/docs/iot/wireless/api/command-resource). Your server should respond with an HTTP status code in the 200 range; any response body will be ignored.
     */
    commandsCallbackUrl: string;
    /**
     * The HTTP method we use to call `commands_callback_url`.  Can be: `POST` or `GET`. Default is `POST`.
     */
    commandsCallbackMethod: string;
    /**
     * Deprecated.
     */
    smsFallbackMethod: string;
    /**
     * Deprecated.
     */
    smsFallbackUrl: string;
    /**
     * Deprecated.
     */
    smsMethod: string;
    /**
     * Deprecated.
     */
    smsUrl: string;
    /**
     * Deprecated. The HTTP method we use to call `voice_fallback_url`. Can be: `GET` or `POST`. Default is `POST`.
     */
    voiceFallbackMethod: string;
    /**
     * Deprecated. The URL we call using the `voice_fallback_method` when an error occurs while retrieving or executing the TwiML requested from `voice_url`.
     */
    voiceFallbackUrl: string;
    /**
     * Deprecated. The HTTP method we use to call `voice_url`. Can be: `GET` or `POST`. Default is `POST`.
     */
    voiceMethod: string;
    /**
     * Deprecated. The URL we call using the `voice_method` when the SIM-connected device makes a voice call.
     */
    voiceUrl: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the Sim resource was last updated specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    /**
     * The URLs of related subresources.
     */
    links: Record<string, string>;
    /**
     * Deprecated.
     */
    ipAddress: string;
    private get _proxy();
    /**
     * Remove a SimInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SimInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SimInstance
     */
    fetch(callback?: (error: Error | null, item?: SimInstance) => any): Promise<SimInstance>;
    /**
     * Update a SimInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SimInstance
     */
    update(callback?: (error: Error | null, item?: SimInstance) => any): Promise<SimInstance>;
    /**
     * Update a SimInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SimInstance
     */
    update(params: SimContextUpdateOptions, callback?: (error: Error | null, item?: SimInstance) => any): Promise<SimInstance>;
    /**
     * Access the dataSessions.
     */
    dataSessions(): DataSessionListInstance;
    /**
     * Access the usageRecords.
     */
    usageRecords(): UsageRecordListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        uniqueName: string;
        accountSid: string;
        ratePlanSid: string;
        friendlyName: string;
        iccid: string;
        eId: string;
        status: SimStatus;
        resetStatus: "resetting";
        commandsCallbackUrl: string;
        commandsCallbackMethod: string;
        smsFallbackMethod: string;
        smsFallbackUrl: string;
        smsMethod: string;
        smsUrl: string;
        voiceFallbackMethod: string;
        voiceFallbackUrl: string;
        voiceMethod: string;
        voiceUrl: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
        ipAddress: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SimSolution {
}
export interface SimListInstance {
    _version: V1;
    _solution: SimSolution;
    _uri: string;
    (sid: string): SimContext;
    get(sid: string): SimContext;
    /**
     * Streams SimInstance records from the API.
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
     * @param { SimListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SimInstance, done: (err?: Error) => void) => void): void;
    each(params: SimListInstanceEachOptions, callback?: (item: SimInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SimInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SimPage) => any): Promise<SimPage>;
    /**
     * Lists SimInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SimListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SimInstance[]) => any): Promise<SimInstance[]>;
    list(params: SimListInstanceOptions, callback?: (error: Error | null, items: SimInstance[]) => any): Promise<SimInstance[]>;
    /**
     * Retrieve a single page of SimInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SimListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SimPage) => any): Promise<SimPage>;
    page(params: SimListInstancePageOptions, callback?: (error: Error | null, items: SimPage) => any): Promise<SimPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SimListInstance(version: V1): SimListInstance;
export declare class SimPage extends Page<V1, SimPayload, SimResource, SimInstance> {
    /**
     * Initialize the SimPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SimSolution);
    /**
     * Build an instance of SimInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SimResource): SimInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
