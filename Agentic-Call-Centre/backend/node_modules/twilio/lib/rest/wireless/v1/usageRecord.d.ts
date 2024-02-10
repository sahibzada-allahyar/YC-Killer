/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type UsageRecordGranularity = "hourly" | "daily" | "all";
/**
 * Options to pass to each
 */
export interface UsageRecordListInstanceEachOptions {
    /** Only include usage that has occurred on or before this date. Format is [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html). */
    end?: Date;
    /** Only include usage that has occurred on or after this date. Format is [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html). */
    start?: Date;
    /** How to summarize the usage by time. Can be: `daily`, `hourly`, or `all`. A value of `all` returns one Usage Record that describes the usage for the entire period. */
    granularity?: UsageRecordGranularity;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: UsageRecordInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface UsageRecordListInstanceOptions {
    /** Only include usage that has occurred on or before this date. Format is [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html). */
    end?: Date;
    /** Only include usage that has occurred on or after this date. Format is [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html). */
    start?: Date;
    /** How to summarize the usage by time. Can be: `daily`, `hourly`, or `all`. A value of `all` returns one Usage Record that describes the usage for the entire period. */
    granularity?: UsageRecordGranularity;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface UsageRecordListInstancePageOptions {
    /** Only include usage that has occurred on or before this date. Format is [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html). */
    end?: Date;
    /** Only include usage that has occurred on or after this date. Format is [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html). */
    start?: Date;
    /** How to summarize the usage by time. Can be: `daily`, `hourly`, or `all`. A value of `all` returns one Usage Record that describes the usage for the entire period. */
    granularity?: UsageRecordGranularity;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface UsageRecordSolution {
}
export interface UsageRecordListInstance {
    _version: V1;
    _solution: UsageRecordSolution;
    _uri: string;
    /**
     * Streams UsageRecordInstance records from the API.
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
     * @param { UsageRecordListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: UsageRecordInstance, done: (err?: Error) => void) => void): void;
    each(params: UsageRecordListInstanceEachOptions, callback?: (item: UsageRecordInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of UsageRecordInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: UsageRecordPage) => any): Promise<UsageRecordPage>;
    /**
     * Lists UsageRecordInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UsageRecordListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: UsageRecordInstance[]) => any): Promise<UsageRecordInstance[]>;
    list(params: UsageRecordListInstanceOptions, callback?: (error: Error | null, items: UsageRecordInstance[]) => any): Promise<UsageRecordInstance[]>;
    /**
     * Retrieve a single page of UsageRecordInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { UsageRecordListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: UsageRecordPage) => any): Promise<UsageRecordPage>;
    page(params: UsageRecordListInstancePageOptions, callback?: (error: Error | null, items: UsageRecordPage) => any): Promise<UsageRecordPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function UsageRecordListInstance(version: V1): UsageRecordListInstance;
interface UsageRecordPayload extends TwilioResponsePayload {
    usage_records: UsageRecordResource[];
}
interface UsageRecordResource {
    account_sid: string;
    period: any;
    commands: any;
    data: any;
}
export declare class UsageRecordInstance {
    protected _version: V1;
    constructor(_version: V1, payload: UsageRecordResource);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the AccountUsageRecord resource.
     */
    accountSid: string;
    /**
     * The time period for which usage is reported. Contains `start` and `end` properties that describe the period using GMT date-time values specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format.
     */
    period: any;
    /**
     * An object that describes the aggregated Commands usage for all SIMs during the specified period. See [Commands Usage Object](https://www.twilio.com/docs/iot/wireless/api/account-usagerecord-resource#commands-usage-object).
     */
    commands: any;
    /**
     * An object that describes the aggregated Data usage for all SIMs over the period. See [Data Usage Object](https://www.twilio.com/docs/iot/wireless/api/account-usagerecord-resource#data-usage-object).
     */
    data: any;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        period: any;
        commands: any;
        data: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class UsageRecordPage extends Page<V1, UsageRecordPayload, UsageRecordResource, UsageRecordInstance> {
    /**
     * Initialize the UsageRecordPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: UsageRecordSolution);
    /**
     * Build an instance of UsageRecordInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: UsageRecordResource): UsageRecordInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
