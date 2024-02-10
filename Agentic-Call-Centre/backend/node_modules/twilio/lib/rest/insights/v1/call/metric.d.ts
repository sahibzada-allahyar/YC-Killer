/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type MetricStreamDirection = "unknown" | "inbound" | "outbound" | "both";
export type MetricTwilioEdge = "unknown_edge" | "carrier_edge" | "sip_edge" | "sdk_edge" | "client_edge";
/**
 * Options to pass to each
 */
export interface MetricListInstanceEachOptions {
    /** The Edge of this Metric. One of `unknown_edge`, `carrier_edge`, `sip_edge`, `sdk_edge` or `client_edge`. */
    edge?: MetricTwilioEdge;
    /** The Direction of this Metric. One of `unknown`, `inbound`, `outbound` or `both`. */
    direction?: MetricStreamDirection;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: MetricInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface MetricListInstanceOptions {
    /** The Edge of this Metric. One of `unknown_edge`, `carrier_edge`, `sip_edge`, `sdk_edge` or `client_edge`. */
    edge?: MetricTwilioEdge;
    /** The Direction of this Metric. One of `unknown`, `inbound`, `outbound` or `both`. */
    direction?: MetricStreamDirection;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface MetricListInstancePageOptions {
    /** The Edge of this Metric. One of `unknown_edge`, `carrier_edge`, `sip_edge`, `sdk_edge` or `client_edge`. */
    edge?: MetricTwilioEdge;
    /** The Direction of this Metric. One of `unknown`, `inbound`, `outbound` or `both`. */
    direction?: MetricStreamDirection;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface MetricSolution {
    callSid: string;
}
export interface MetricListInstance {
    _version: V1;
    _solution: MetricSolution;
    _uri: string;
    /**
     * Streams MetricInstance records from the API.
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
     * @param { MetricListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: MetricInstance, done: (err?: Error) => void) => void): void;
    each(params: MetricListInstanceEachOptions, callback?: (item: MetricInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of MetricInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: MetricPage) => any): Promise<MetricPage>;
    /**
     * Lists MetricInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MetricListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: MetricInstance[]) => any): Promise<MetricInstance[]>;
    list(params: MetricListInstanceOptions, callback?: (error: Error | null, items: MetricInstance[]) => any): Promise<MetricInstance[]>;
    /**
     * Retrieve a single page of MetricInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MetricListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: MetricPage) => any): Promise<MetricPage>;
    page(params: MetricListInstancePageOptions, callback?: (error: Error | null, items: MetricPage) => any): Promise<MetricPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function MetricListInstance(version: V1, callSid: string): MetricListInstance;
interface MetricPayload extends TwilioResponsePayload {
    metrics: MetricResource[];
}
interface MetricResource {
    timestamp: string;
    call_sid: string;
    account_sid: string;
    edge: MetricTwilioEdge;
    direction: MetricStreamDirection;
    carrier_edge: any;
    sip_edge: any;
    sdk_edge: any;
    client_edge: any;
}
export declare class MetricInstance {
    protected _version: V1;
    constructor(_version: V1, payload: MetricResource, callSid: string);
    /**
     * Timestamp of metric sample. Samples are taken every 10 seconds and contain the metrics for the previous 10 seconds.
     */
    timestamp: string;
    /**
     * The unique SID identifier of the Call.
     */
    callSid: string;
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    edge: MetricTwilioEdge;
    direction: MetricStreamDirection;
    /**
     * Contains metrics and properties for the Twilio media gateway of a PSTN call.
     */
    carrierEdge: any;
    /**
     * Contains metrics and properties for the Twilio media gateway of a SIP Interface or Trunking call.
     */
    sipEdge: any;
    /**
     * Contains metrics and properties for the SDK sensor library for Client calls.
     */
    sdkEdge: any;
    /**
     * Contains metrics and properties for the Twilio media gateway of a Client call.
     */
    clientEdge: any;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        timestamp: string;
        callSid: string;
        accountSid: string;
        edge: MetricTwilioEdge;
        direction: MetricStreamDirection;
        carrierEdge: any;
        sipEdge: any;
        sdkEdge: any;
        clientEdge: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class MetricPage extends Page<V1, MetricPayload, MetricResource, MetricInstance> {
    /**
     * Initialize the MetricPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: MetricSolution);
    /**
     * Build an instance of MetricInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: MetricResource): MetricInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
