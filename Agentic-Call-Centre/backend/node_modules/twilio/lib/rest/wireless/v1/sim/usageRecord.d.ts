/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type UsageRecordGranularity = "hourly" | "daily" | "all";
/**
 * Options to pass to each
 */
export interface UsageRecordListInstanceEachOptions {
    /** Only include usage that occurred on or before this date, specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html). The default is the current time. */
    end?: Date;
    /** Only include usage that has occurred on or after this date, specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html). The default is one month before the `end` parameter value. */
    start?: Date;
    /** How to summarize the usage by time. Can be: `daily`, `hourly`, or `all`. The default is `all`. A value of `all` returns one Usage Record that describes the usage for the entire period. */
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
    /** Only include usage that occurred on or before this date, specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html). The default is the current time. */
    end?: Date;
    /** Only include usage that has occurred on or after this date, specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html). The default is one month before the `end` parameter value. */
    start?: Date;
    /** How to summarize the usage by time. Can be: `daily`, `hourly`, or `all`. The default is `all`. A value of `all` returns one Usage Record that describes the usage for the entire period. */
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
    /** Only include usage that occurred on or before this date, specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html). The default is the current time. */
    end?: Date;
    /** Only include usage that has occurred on or after this date, specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html). The default is one month before the `end` parameter value. */
    start?: Date;
    /** How to summarize the usage by time. Can be: `daily`, `hourly`, or `all`. The default is `all`. A value of `all` returns one Usage Record that describes the usage for the entire period. */
    granularity?: UsageRecordGranularity;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface UsageRecordSolution {
    simSid: string;
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
export declare function UsageRecordListInstance(version: V1, simSid: string): UsageRecordListInstance;
interface UsageRecordPayload extends TwilioResponsePayload {
    usage_records: UsageRecordResource[];
}
interface UsageRecordResource {
    sim_sid: string;
    account_sid: string;
    period: any;
    commands: any;
    data: any;
}
export declare class UsageRecordInstance {
    protected _version: V1;
    constructor(_version: V1, payload: UsageRecordResource, simSid: string);
    /**
     * The SID of the [Sim resource](https://www.twilio.com/docs/iot/wireless/api/sim-resource) that this Usage Record is for.
     */
    simSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the UsageRecord resource.
     */
    accountSid: string;
    /**
     * The time period for which the usage is reported. Contains `start` and `end` datetime values given as GMT in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format.
     */
    period: any;
    /**
     * An object that describes the SIM\'s usage of Commands during the specified period. See [Commands Usage Object](https://www.twilio.com/docs/iot/wireless/api/sim-usagerecord-resource#commands-usage-object).
     */
    commands: any;
    /**
     * An object that describes the SIM\'s data usage during the specified period. See [Data Usage Object](https://www.twilio.com/docs/iot/wireless/api/sim-usagerecord-resource#data-usage-object).
     */
    data: any;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        simSid: string;
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
