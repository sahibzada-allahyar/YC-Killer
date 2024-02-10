/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type UsageRecordGranularity = "hour" | "day" | "all";
export type UsageRecordGroup = "sim" | "fleet" | "network" | "isoCountry";
/**
 * Options to pass to each
 */
export interface UsageRecordListInstanceEachOptions {
    /** SID or unique name of a Sim resource. Only show UsageRecords representing usage incurred by this Super SIM. */
    sim?: string;
    /** SID or unique name of a Fleet resource. Only show UsageRecords representing usage for Super SIMs belonging to this Fleet resource at the time the usage occurred. */
    fleet?: string;
    /** SID of a Network resource. Only show UsageRecords representing usage on this network. */
    network?: string;
    /** Alpha-2 ISO Country Code. Only show UsageRecords representing usage in this country. */
    isoCountry?: string;
    /** Dimension over which to aggregate usage records. Can be: `sim`, `fleet`, `network`, `isoCountry`. Default is to not aggregate across any of these dimensions, UsageRecords will be aggregated into the time buckets described by the `Granularity` parameter. */
    group?: UsageRecordGroup;
    /** Time-based grouping that UsageRecords should be aggregated by. Can be: `hour`, `day`, or `all`. Default is `all`. `all` returns one UsageRecord that describes the usage for the entire period. */
    granularity?: UsageRecordGranularity;
    /** Only include usage that occurred at or after this time, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. Default is one month before the `end_time`. */
    startTime?: Date;
    /** Only include usage that occurred before this time (exclusive), specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. Default is the current time. */
    endTime?: Date;
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
    /** SID or unique name of a Sim resource. Only show UsageRecords representing usage incurred by this Super SIM. */
    sim?: string;
    /** SID or unique name of a Fleet resource. Only show UsageRecords representing usage for Super SIMs belonging to this Fleet resource at the time the usage occurred. */
    fleet?: string;
    /** SID of a Network resource. Only show UsageRecords representing usage on this network. */
    network?: string;
    /** Alpha-2 ISO Country Code. Only show UsageRecords representing usage in this country. */
    isoCountry?: string;
    /** Dimension over which to aggregate usage records. Can be: `sim`, `fleet`, `network`, `isoCountry`. Default is to not aggregate across any of these dimensions, UsageRecords will be aggregated into the time buckets described by the `Granularity` parameter. */
    group?: UsageRecordGroup;
    /** Time-based grouping that UsageRecords should be aggregated by. Can be: `hour`, `day`, or `all`. Default is `all`. `all` returns one UsageRecord that describes the usage for the entire period. */
    granularity?: UsageRecordGranularity;
    /** Only include usage that occurred at or after this time, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. Default is one month before the `end_time`. */
    startTime?: Date;
    /** Only include usage that occurred before this time (exclusive), specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. Default is the current time. */
    endTime?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface UsageRecordListInstancePageOptions {
    /** SID or unique name of a Sim resource. Only show UsageRecords representing usage incurred by this Super SIM. */
    sim?: string;
    /** SID or unique name of a Fleet resource. Only show UsageRecords representing usage for Super SIMs belonging to this Fleet resource at the time the usage occurred. */
    fleet?: string;
    /** SID of a Network resource. Only show UsageRecords representing usage on this network. */
    network?: string;
    /** Alpha-2 ISO Country Code. Only show UsageRecords representing usage in this country. */
    isoCountry?: string;
    /** Dimension over which to aggregate usage records. Can be: `sim`, `fleet`, `network`, `isoCountry`. Default is to not aggregate across any of these dimensions, UsageRecords will be aggregated into the time buckets described by the `Granularity` parameter. */
    group?: UsageRecordGroup;
    /** Time-based grouping that UsageRecords should be aggregated by. Can be: `hour`, `day`, or `all`. Default is `all`. `all` returns one UsageRecord that describes the usage for the entire period. */
    granularity?: UsageRecordGranularity;
    /** Only include usage that occurred at or after this time, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. Default is one month before the `end_time`. */
    startTime?: Date;
    /** Only include usage that occurred before this time (exclusive), specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. Default is the current time. */
    endTime?: Date;
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
    sim_sid: string;
    network_sid: string;
    fleet_sid: string;
    iso_country: string;
    period: any;
    data_upload: number;
    data_download: number;
    data_total: number;
    data_total_billed: number;
    billed_unit: string;
}
export declare class UsageRecordInstance {
    protected _version: V1;
    constructor(_version: V1, payload: UsageRecordResource);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that incurred the usage.
     */
    accountSid: string;
    /**
     * SID of a Sim resource to which the UsageRecord belongs. Value will only be present when either a value for the `Sim` query parameter is provided or when UsageRecords are grouped by `sim`. Otherwise, the value will be `null`.
     */
    simSid: string;
    /**
     * SID of the Network resource the usage occurred on. Value will only be present when either a value for the `Network` query parameter is provided or when UsageRecords are grouped by `network`. Otherwise, the value will be `null`.
     */
    networkSid: string;
    /**
     * SID of the Fleet resource the usage occurred on. Value will only be present when either a value for the `Fleet` query parameter is provided or when UsageRecords are grouped by `fleet`. Otherwise, the value will be `null`.
     */
    fleetSid: string;
    /**
     * Alpha-2 ISO Country Code that the usage occurred in. Value will only be present when either a value for the `IsoCountry` query parameter is provided or when UsageRecords are grouped by `isoCountry`. Otherwise, the value will be `null`.
     */
    isoCountry: string;
    /**
     * The time period for which the usage is reported. The period is represented as a pair of `start_time` and `end_time` timestamps specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    period: any;
    /**
     * Total data uploaded in bytes, aggregated by the query parameters.
     */
    dataUpload: number;
    /**
     * Total data downloaded in bytes, aggregated by the query parameters.
     */
    dataDownload: number;
    /**
     * Total of data_upload and data_download.
     */
    dataTotal: number;
    /**
     * Total amount in the `billed_unit` that was charged for the data uploaded or downloaded. Will return 0 for usage prior to February 1, 2022. Value may be 0 despite `data_total` being greater than 0 if the data usage is still being processed by Twilio\'s billing system. Refer to [Data Usage Processing](https://www.twilio.com/docs/iot/supersim/api/usage-record-resource#data-usage-processing) for more details.
     */
    dataTotalBilled: number;
    /**
     * The currency in which the billed amounts are measured, specified in the 3 letter ISO 4127 format (e.g. `USD`, `EUR`, `JPY`). This can be null when data_toal_billed is 0 and we do not yet have billing information for the corresponding data usage. Refer to [Data Usage Processing](https://www.twilio.com/docs/iot/supersim/api/usage-record-resource#data-usage-processing) for more details.
     */
    billedUnit: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        simSid: string;
        networkSid: string;
        fleetSid: string;
        isoCountry: string;
        period: any;
        dataUpload: number;
        dataDownload: number;
        dataTotal: number;
        dataTotalBilled: number;
        billedUnit: string;
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
