/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import Wireless from "../Wireless";
import { UsageListInstance } from "./sim/usage";
/**
 * Options to pass to update a SimInstance
 */
export interface SimContextUpdateOptions {
    /**  */
    uniqueName?: string;
    /**  */
    callbackMethod?: string;
    /**  */
    callbackUrl?: string;
    /**  */
    friendlyName?: string;
    /**  */
    ratePlan?: string;
    /**  */
    status?: string;
    /**  */
    commandsCallbackMethod?: string;
    /**  */
    commandsCallbackUrl?: string;
    /**  */
    smsFallbackMethod?: string;
    /**  */
    smsFallbackUrl?: string;
    /**  */
    smsMethod?: string;
    /**  */
    smsUrl?: string;
    /**  */
    voiceFallbackMethod?: string;
    /**  */
    voiceFallbackUrl?: string;
    /**  */
    voiceMethod?: string;
    /**  */
    voiceUrl?: string;
}
/**
 * Options to pass to each
 */
export interface SimListInstanceEachOptions {
    /**  */
    status?: string;
    /**  */
    iccid?: string;
    /**  */
    ratePlan?: string;
    /**  */
    eId?: string;
    /**  */
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
    /**  */
    status?: string;
    /**  */
    iccid?: string;
    /**  */
    ratePlan?: string;
    /**  */
    eId?: string;
    /**  */
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
    /**  */
    status?: string;
    /**  */
    iccid?: string;
    /**  */
    ratePlan?: string;
    /**  */
    eId?: string;
    /**  */
    simRegistrationCode?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SimContext {
    usage: UsageListInstance;
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
    protected _version: Wireless;
    protected _solution: SimContextSolution;
    protected _uri: string;
    protected _usage?: UsageListInstance;
    constructor(_version: Wireless, sid: string);
    get usage(): UsageListInstance;
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
    status: string;
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
}
export declare class SimInstance {
    protected _version: Wireless;
    protected _solution: SimContextSolution;
    protected _context?: SimContext;
    constructor(_version: Wireless, payload: SimResource, sid?: string);
    sid: string;
    uniqueName: string;
    accountSid: string;
    ratePlanSid: string;
    friendlyName: string;
    iccid: string;
    eId: string;
    status: string;
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
    private get _proxy();
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
     * Access the usage.
     */
    usage(): UsageListInstance;
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
        status: string;
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
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SimSolution {
}
export interface SimListInstance {
    _version: Wireless;
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
export declare function SimListInstance(version: Wireless): SimListInstance;
export declare class SimPage extends Page<Wireless, SimPayload, SimResource, SimInstance> {
    /**
     * Initialize the SimPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: Wireless, response: Response<string>, solution: SimSolution);
    /**
     * Build an instance of SimInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SimResource): SimInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
