/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import Wireless from "../Wireless";
/**
 * Options to pass to update a RatePlanInstance
 */
export interface RatePlanContextUpdateOptions {
    /**  */
    uniqueName?: string;
    /**  */
    friendlyName?: string;
}
/**
 * Options to pass to create a RatePlanInstance
 */
export interface RatePlanListInstanceCreateOptions {
    /**  */
    uniqueName?: string;
    /**  */
    friendlyName?: string;
    /**  */
    dataEnabled?: boolean;
    /**  */
    dataLimit?: number;
    /**  */
    dataMetering?: string;
    /**  */
    messagingEnabled?: boolean;
    /**  */
    voiceEnabled?: boolean;
    /**  */
    commandsEnabled?: boolean;
    /**  */
    nationalRoamingEnabled?: boolean;
    /**  */
    internationalRoaming?: Array<string>;
}
/**
 * Options to pass to each
 */
export interface RatePlanListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: RatePlanInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface RatePlanListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface RatePlanListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface RatePlanContext {
    /**
     * Remove a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    fetch(callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Update a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    update(callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Update a RatePlanInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    update(params: RatePlanContextUpdateOptions, callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface RatePlanContextSolution {
    sid: string;
}
export declare class RatePlanContextImpl implements RatePlanContext {
    protected _version: Wireless;
    protected _solution: RatePlanContextSolution;
    protected _uri: string;
    constructor(_version: Wireless, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    update(params?: RatePlanContextUpdateOptions | ((error: Error | null, item?: RatePlanInstance) => any), callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): RatePlanContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface RatePlanPayload extends TwilioResponsePayload {
    rate_plans: RatePlanResource[];
}
interface RatePlanResource {
    sid: string;
    unique_name: string;
    account_sid: string;
    friendly_name: string;
    data_enabled: boolean;
    data_metering: string;
    data_limit: number;
    messaging_enabled: boolean;
    voice_enabled: boolean;
    national_roaming_enabled: boolean;
    international_roaming: Array<string>;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class RatePlanInstance {
    protected _version: Wireless;
    protected _solution: RatePlanContextSolution;
    protected _context?: RatePlanContext;
    constructor(_version: Wireless, payload: RatePlanResource, sid?: string);
    sid: string;
    uniqueName: string;
    accountSid: string;
    friendlyName: string;
    dataEnabled: boolean;
    dataMetering: string;
    dataLimit: number;
    messagingEnabled: boolean;
    voiceEnabled: boolean;
    nationalRoamingEnabled: boolean;
    internationalRoaming: Array<string>;
    dateCreated: Date;
    dateUpdated: Date;
    url: string;
    private get _proxy();
    /**
     * Remove a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    fetch(callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Update a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    update(callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Update a RatePlanInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    update(params: RatePlanContextUpdateOptions, callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        uniqueName: string;
        accountSid: string;
        friendlyName: string;
        dataEnabled: boolean;
        dataMetering: string;
        dataLimit: number;
        messagingEnabled: boolean;
        voiceEnabled: boolean;
        nationalRoamingEnabled: boolean;
        internationalRoaming: string[];
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface RatePlanSolution {
}
export interface RatePlanListInstance {
    _version: Wireless;
    _solution: RatePlanSolution;
    _uri: string;
    (sid: string): RatePlanContext;
    get(sid: string): RatePlanContext;
    /**
     * Create a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    create(callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Create a RatePlanInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    create(params: RatePlanListInstanceCreateOptions, callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Streams RatePlanInstance records from the API.
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
     * @param { RatePlanListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: RatePlanInstance, done: (err?: Error) => void) => void): void;
    each(params: RatePlanListInstanceEachOptions, callback?: (item: RatePlanInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of RatePlanInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: RatePlanPage) => any): Promise<RatePlanPage>;
    /**
     * Lists RatePlanInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RatePlanListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: RatePlanInstance[]) => any): Promise<RatePlanInstance[]>;
    list(params: RatePlanListInstanceOptions, callback?: (error: Error | null, items: RatePlanInstance[]) => any): Promise<RatePlanInstance[]>;
    /**
     * Retrieve a single page of RatePlanInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RatePlanListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: RatePlanPage) => any): Promise<RatePlanPage>;
    page(params: RatePlanListInstancePageOptions, callback?: (error: Error | null, items: RatePlanPage) => any): Promise<RatePlanPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RatePlanListInstance(version: Wireless): RatePlanListInstance;
export declare class RatePlanPage extends Page<Wireless, RatePlanPayload, RatePlanResource, RatePlanInstance> {
    /**
     * Initialize the RatePlanPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: Wireless, response: Response<string>, solution: RatePlanSolution);
    /**
     * Build an instance of RatePlanInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: RatePlanResource): RatePlanInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
