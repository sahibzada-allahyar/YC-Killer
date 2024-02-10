/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { BillingPeriodListInstance } from "./sim/billingPeriod";
import { SimIpAddressListInstance } from "./sim/simIpAddress";
export type SimStatus = "new" | "ready" | "active" | "inactive" | "scheduled";
export type SimStatusUpdate = "ready" | "active" | "inactive";
/**
 * Options to pass to update a SimInstance
 */
export interface SimContextUpdateOptions {
    /** An application-defined string that uniquely identifies the resource. It can be used in place of the resource\\\'s `sid` in the URL to address the resource. */
    uniqueName?: string;
    /**  */
    status?: SimStatusUpdate;
    /** The SID or unique name of the Fleet to which the SIM resource should be assigned. */
    fleet?: string;
    /** The URL we should call using the `callback_method` after an asynchronous update has finished. */
    callbackUrl?: string;
    /** The HTTP method we should use to call `callback_url`. Can be: `GET` or `POST` and the default is POST. */
    callbackMethod?: string;
    /** The SID of the Account to which the Sim resource should belong. The Account SID can only be that of the requesting Account or that of a Subaccount of the requesting Account. Only valid when the Sim resource\\\'s status is new. */
    accountSid?: string;
}
/**
 * Options to pass to create a SimInstance
 */
export interface SimListInstanceCreateOptions {
    /** The [ICCID](https://en.wikipedia.org/wiki/Subscriber_identity_module#ICCID) of the Super SIM to be added to your Account. */
    iccid: string;
    /** The 10-digit code required to claim the Super SIM for your Account. */
    registrationCode: string;
}
/**
 * Options to pass to each
 */
export interface SimListInstanceEachOptions {
    /** The status of the Sim resources to read. Can be `new`, `ready`, `active`, `inactive`, or `scheduled`. */
    status?: SimStatus;
    /** The SID or unique name of the Fleet to which a list of Sims are assigned. */
    fleet?: string;
    /** The [ICCID](https://en.wikipedia.org/wiki/Subscriber_identity_module#ICCID) associated with a Super SIM to filter the list by. Passing this parameter will always return a list containing zero or one SIMs. */
    iccid?: string;
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
    /** The status of the Sim resources to read. Can be `new`, `ready`, `active`, `inactive`, or `scheduled`. */
    status?: SimStatus;
    /** The SID or unique name of the Fleet to which a list of Sims are assigned. */
    fleet?: string;
    /** The [ICCID](https://en.wikipedia.org/wiki/Subscriber_identity_module#ICCID) associated with a Super SIM to filter the list by. Passing this parameter will always return a list containing zero or one SIMs. */
    iccid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SimListInstancePageOptions {
    /** The status of the Sim resources to read. Can be `new`, `ready`, `active`, `inactive`, or `scheduled`. */
    status?: SimStatus;
    /** The SID or unique name of the Fleet to which a list of Sims are assigned. */
    fleet?: string;
    /** The [ICCID](https://en.wikipedia.org/wiki/Subscriber_identity_module#ICCID) associated with a Super SIM to filter the list by. Passing this parameter will always return a list containing zero or one SIMs. */
    iccid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SimContext {
    billingPeriods: BillingPeriodListInstance;
    simIpAddresses: SimIpAddressListInstance;
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
    protected _billingPeriods?: BillingPeriodListInstance;
    protected _simIpAddresses?: SimIpAddressListInstance;
    constructor(_version: V1, sid: string);
    get billingPeriods(): BillingPeriodListInstance;
    get simIpAddresses(): SimIpAddressListInstance;
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
    iccid: string;
    status: SimStatus;
    fleet_sid: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class SimInstance {
    protected _version: V1;
    protected _solution: SimContextSolution;
    protected _context?: SimContext;
    constructor(_version: V1, payload: SimResource, sid?: string);
    /**
     * The unique string that identifies the Sim resource.
     */
    sid: string;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used in place of the resource\'s `sid` in the URL to address the resource.
     */
    uniqueName: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that the Super SIM belongs to.
     */
    accountSid: string;
    /**
     * The [ICCID](https://en.wikipedia.org/wiki/Subscriber_identity_module#ICCID) associated with the SIM.
     */
    iccid: string;
    status: SimStatus;
    /**
     * The unique ID of the Fleet configured for this SIM.
     */
    fleetSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Sim Resource.
     */
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
     * Access the billingPeriods.
     */
    billingPeriods(): BillingPeriodListInstance;
    /**
     * Access the simIpAddresses.
     */
    simIpAddresses(): SimIpAddressListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        uniqueName: string;
        accountSid: string;
        iccid: string;
        status: SimStatus;
        fleetSid: string;
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
    _version: V1;
    _solution: SimSolution;
    _uri: string;
    (sid: string): SimContext;
    get(sid: string): SimContext;
    /**
     * Create a SimInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SimInstance
     */
    create(params: SimListInstanceCreateOptions, callback?: (error: Error | null, item?: SimInstance) => any): Promise<SimInstance>;
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
