/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type BillingPeriodBpType = "ready" | "active";
/**
 * Options to pass to each
 */
export interface BillingPeriodListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: BillingPeriodInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface BillingPeriodListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface BillingPeriodListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface BillingPeriodSolution {
    simSid: string;
}
export interface BillingPeriodListInstance {
    _version: V1;
    _solution: BillingPeriodSolution;
    _uri: string;
    /**
     * Streams BillingPeriodInstance records from the API.
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
     * @param { BillingPeriodListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: BillingPeriodInstance, done: (err?: Error) => void) => void): void;
    each(params: BillingPeriodListInstanceEachOptions, callback?: (item: BillingPeriodInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of BillingPeriodInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: BillingPeriodPage) => any): Promise<BillingPeriodPage>;
    /**
     * Lists BillingPeriodInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BillingPeriodListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: BillingPeriodInstance[]) => any): Promise<BillingPeriodInstance[]>;
    list(params: BillingPeriodListInstanceOptions, callback?: (error: Error | null, items: BillingPeriodInstance[]) => any): Promise<BillingPeriodInstance[]>;
    /**
     * Retrieve a single page of BillingPeriodInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BillingPeriodListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: BillingPeriodPage) => any): Promise<BillingPeriodPage>;
    page(params: BillingPeriodListInstancePageOptions, callback?: (error: Error | null, items: BillingPeriodPage) => any): Promise<BillingPeriodPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BillingPeriodListInstance(version: V1, simSid: string): BillingPeriodListInstance;
interface BillingPeriodPayload extends TwilioResponsePayload {
    billing_periods: BillingPeriodResource[];
}
interface BillingPeriodResource {
    sid: string;
    account_sid: string;
    sim_sid: string;
    start_time: Date;
    end_time: Date;
    period_type: BillingPeriodBpType;
    date_created: Date;
    date_updated: Date;
}
export declare class BillingPeriodInstance {
    protected _version: V1;
    constructor(_version: V1, payload: BillingPeriodResource, simSid: string);
    /**
     * The SID of the Billing Period.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) the Super SIM belongs to.
     */
    accountSid: string;
    /**
     * The SID of the Super SIM the Billing Period belongs to.
     */
    simSid: string;
    /**
     * The start time of the Billing Period specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    startTime: Date;
    /**
     * The end time of the Billing Period specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    endTime: Date;
    periodType: BillingPeriodBpType;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        simSid: string;
        startTime: Date;
        endTime: Date;
        periodType: BillingPeriodBpType;
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class BillingPeriodPage extends Page<V1, BillingPeriodPayload, BillingPeriodResource, BillingPeriodInstance> {
    /**
     * Initialize the BillingPeriodPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: BillingPeriodSolution);
    /**
     * Build an instance of BillingPeriodInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: BillingPeriodResource): BillingPeriodInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
