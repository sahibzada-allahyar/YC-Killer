/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { EngagementListInstance } from "./flow/engagement";
import { ExecutionListInstance } from "./flow/execution";
export type FlowStatus = "draft" | "published";
/**
 * Options to pass to each
 */
export interface FlowListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: FlowInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface FlowListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface FlowListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface FlowContext {
    engagements: EngagementListInstance;
    executions: ExecutionListInstance;
    /**
     * Remove a FlowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a FlowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlowInstance
     */
    fetch(callback?: (error: Error | null, item?: FlowInstance) => any): Promise<FlowInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface FlowContextSolution {
    sid: string;
}
export declare class FlowContextImpl implements FlowContext {
    protected _version: V1;
    protected _solution: FlowContextSolution;
    protected _uri: string;
    protected _engagements?: EngagementListInstance;
    protected _executions?: ExecutionListInstance;
    constructor(_version: V1, sid: string);
    get engagements(): EngagementListInstance;
    get executions(): ExecutionListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: FlowInstance) => any): Promise<FlowInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): FlowContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface FlowPayload extends TwilioResponsePayload {
    flows: FlowResource[];
}
interface FlowResource {
    sid: string;
    account_sid: string;
    friendly_name: string;
    status: FlowStatus;
    version: number;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class FlowInstance {
    protected _version: V1;
    protected _solution: FlowContextSolution;
    protected _context?: FlowContext;
    constructor(_version: V1, payload: FlowResource, sid?: string);
    /**
     * The unique string that we created to identify the Flow resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Flow resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the Flow.
     */
    friendlyName: string;
    status: FlowStatus;
    /**
     * The latest version number of the Flow\'s definition.
     */
    version: number;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    /**
     * The URLs of the Flow\'s nested resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a FlowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a FlowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlowInstance
     */
    fetch(callback?: (error: Error | null, item?: FlowInstance) => any): Promise<FlowInstance>;
    /**
     * Access the engagements.
     */
    engagements(): EngagementListInstance;
    /**
     * Access the executions.
     */
    executions(): ExecutionListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        friendlyName: string;
        status: FlowStatus;
        version: number;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface FlowSolution {
}
export interface FlowListInstance {
    _version: V1;
    _solution: FlowSolution;
    _uri: string;
    (sid: string): FlowContext;
    get(sid: string): FlowContext;
    /**
     * Streams FlowInstance records from the API.
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
     * @param { FlowListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: FlowInstance, done: (err?: Error) => void) => void): void;
    each(params: FlowListInstanceEachOptions, callback?: (item: FlowInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of FlowInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: FlowPage) => any): Promise<FlowPage>;
    /**
     * Lists FlowInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FlowListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: FlowInstance[]) => any): Promise<FlowInstance[]>;
    list(params: FlowListInstanceOptions, callback?: (error: Error | null, items: FlowInstance[]) => any): Promise<FlowInstance[]>;
    /**
     * Retrieve a single page of FlowInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FlowListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: FlowPage) => any): Promise<FlowPage>;
    page(params: FlowListInstancePageOptions, callback?: (error: Error | null, items: FlowPage) => any): Promise<FlowPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function FlowListInstance(version: V1): FlowListInstance;
export declare class FlowPage extends Page<V1, FlowPayload, FlowResource, FlowInstance> {
    /**
     * Initialize the FlowPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: FlowSolution);
    /**
     * Build an instance of FlowInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: FlowResource): FlowInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
