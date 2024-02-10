/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { SinkTestListInstance } from "./sink/sinkTest";
import { SinkValidateListInstance } from "./sink/sinkValidate";
export type SinkSinkType = "kinesis" | "webhook" | "segment";
export type SinkStatus = "initialized" | "validating" | "active" | "failed";
/**
 * Options to pass to update a SinkInstance
 */
export interface SinkContextUpdateOptions {
    /** A human readable description for the Sink **This value should not contain PII.** */
    description: string;
}
/**
 * Options to pass to create a SinkInstance
 */
export interface SinkListInstanceCreateOptions {
    /** A human readable description for the Sink **This value should not contain PII.** */
    description: string;
    /** The information required for Twilio to connect to the provided Sink encoded as JSON. */
    sinkConfiguration: any;
    /**  */
    sinkType: SinkSinkType;
}
/**
 * Options to pass to each
 */
export interface SinkListInstanceEachOptions {
    /** A boolean query parameter filtering the results to return sinks used/not used by a subscription. */
    inUse?: boolean;
    /** A String query parameter filtering the results by status `initialized`, `validating`, `active` or `failed`. */
    status?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SinkInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SinkListInstanceOptions {
    /** A boolean query parameter filtering the results to return sinks used/not used by a subscription. */
    inUse?: boolean;
    /** A String query parameter filtering the results by status `initialized`, `validating`, `active` or `failed`. */
    status?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SinkListInstancePageOptions {
    /** A boolean query parameter filtering the results to return sinks used/not used by a subscription. */
    inUse?: boolean;
    /** A String query parameter filtering the results by status `initialized`, `validating`, `active` or `failed`. */
    status?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SinkContext {
    sinkTest: SinkTestListInstance;
    sinkValidate: SinkValidateListInstance;
    /**
     * Remove a SinkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SinkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SinkInstance
     */
    fetch(callback?: (error: Error | null, item?: SinkInstance) => any): Promise<SinkInstance>;
    /**
     * Update a SinkInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SinkInstance
     */
    update(params: SinkContextUpdateOptions, callback?: (error: Error | null, item?: SinkInstance) => any): Promise<SinkInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SinkContextSolution {
    sid: string;
}
export declare class SinkContextImpl implements SinkContext {
    protected _version: V1;
    protected _solution: SinkContextSolution;
    protected _uri: string;
    protected _sinkTest?: SinkTestListInstance;
    protected _sinkValidate?: SinkValidateListInstance;
    constructor(_version: V1, sid: string);
    get sinkTest(): SinkTestListInstance;
    get sinkValidate(): SinkValidateListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SinkInstance) => any): Promise<SinkInstance>;
    update(params: SinkContextUpdateOptions, callback?: (error: Error | null, item?: SinkInstance) => any): Promise<SinkInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SinkContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SinkPayload extends TwilioResponsePayload {
    sinks: SinkResource[];
}
interface SinkResource {
    date_created: Date;
    date_updated: Date;
    description: string;
    sid: string;
    sink_configuration: any;
    sink_type: SinkSinkType;
    status: SinkStatus;
    url: string;
    links: Record<string, string>;
}
export declare class SinkInstance {
    protected _version: V1;
    protected _solution: SinkContextSolution;
    protected _context?: SinkContext;
    constructor(_version: V1, payload: SinkResource, sid?: string);
    /**
     * The date that this Sink was created, given in ISO 8601 format.
     */
    dateCreated: Date;
    /**
     * The date that this Sink was updated, given in ISO 8601 format.
     */
    dateUpdated: Date;
    /**
     * A human readable description for the Sink
     */
    description: string;
    /**
     * A 34 character string that uniquely identifies this Sink.
     */
    sid: string;
    /**
     * The information required for Twilio to connect to the provided Sink encoded as JSON.
     */
    sinkConfiguration: any;
    sinkType: SinkSinkType;
    status: SinkStatus;
    /**
     * The URL of this resource.
     */
    url: string;
    /**
     * Contains a dictionary of URL links to nested resources of this Sink.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a SinkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SinkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SinkInstance
     */
    fetch(callback?: (error: Error | null, item?: SinkInstance) => any): Promise<SinkInstance>;
    /**
     * Update a SinkInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SinkInstance
     */
    update(params: SinkContextUpdateOptions, callback?: (error: Error | null, item?: SinkInstance) => any): Promise<SinkInstance>;
    /**
     * Access the sinkTest.
     */
    sinkTest(): SinkTestListInstance;
    /**
     * Access the sinkValidate.
     */
    sinkValidate(): SinkValidateListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        dateCreated: Date;
        dateUpdated: Date;
        description: string;
        sid: string;
        sinkConfiguration: any;
        sinkType: SinkSinkType;
        status: SinkStatus;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SinkSolution {
}
export interface SinkListInstance {
    _version: V1;
    _solution: SinkSolution;
    _uri: string;
    (sid: string): SinkContext;
    get(sid: string): SinkContext;
    /**
     * Create a SinkInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SinkInstance
     */
    create(params: SinkListInstanceCreateOptions, callback?: (error: Error | null, item?: SinkInstance) => any): Promise<SinkInstance>;
    /**
     * Streams SinkInstance records from the API.
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
     * @param { SinkListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SinkInstance, done: (err?: Error) => void) => void): void;
    each(params: SinkListInstanceEachOptions, callback?: (item: SinkInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SinkInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SinkPage) => any): Promise<SinkPage>;
    /**
     * Lists SinkInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SinkListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SinkInstance[]) => any): Promise<SinkInstance[]>;
    list(params: SinkListInstanceOptions, callback?: (error: Error | null, items: SinkInstance[]) => any): Promise<SinkInstance[]>;
    /**
     * Retrieve a single page of SinkInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SinkListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SinkPage) => any): Promise<SinkPage>;
    page(params: SinkListInstancePageOptions, callback?: (error: Error | null, items: SinkPage) => any): Promise<SinkPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SinkListInstance(version: V1): SinkListInstance;
export declare class SinkPage extends Page<V1, SinkPayload, SinkResource, SinkInstance> {
    /**
     * Initialize the SinkPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SinkSolution);
    /**
     * Build an instance of SinkInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SinkResource): SinkInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
