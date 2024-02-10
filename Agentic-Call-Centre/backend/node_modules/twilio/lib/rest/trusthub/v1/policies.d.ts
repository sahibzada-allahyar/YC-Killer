/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to each
 */
export interface PoliciesListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: PoliciesInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface PoliciesListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface PoliciesListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface PoliciesContext {
    /**
     * Fetch a PoliciesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PoliciesInstance
     */
    fetch(callback?: (error: Error | null, item?: PoliciesInstance) => any): Promise<PoliciesInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PoliciesContextSolution {
    sid: string;
}
export declare class PoliciesContextImpl implements PoliciesContext {
    protected _version: V1;
    protected _solution: PoliciesContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: PoliciesInstance) => any): Promise<PoliciesInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PoliciesContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PoliciesPayload extends TwilioResponsePayload {
    results: PoliciesResource[];
}
interface PoliciesResource {
    sid: string;
    friendly_name: string;
    requirements: any;
    url: string;
}
export declare class PoliciesInstance {
    protected _version: V1;
    protected _solution: PoliciesContextSolution;
    protected _context?: PoliciesContext;
    constructor(_version: V1, payload: PoliciesResource, sid?: string);
    /**
     * The unique string that identifies the Policy resource.
     */
    sid: string;
    /**
     * A human-readable description that is assigned to describe the Policy resource. Examples can include Primary Customer profile policy
     */
    friendlyName: string;
    /**
     * The SID of an object that holds the policy information
     */
    requirements: any;
    /**
     * The absolute URL of the Policy resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a PoliciesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PoliciesInstance
     */
    fetch(callback?: (error: Error | null, item?: PoliciesInstance) => any): Promise<PoliciesInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        friendlyName: string;
        requirements: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PoliciesSolution {
}
export interface PoliciesListInstance {
    _version: V1;
    _solution: PoliciesSolution;
    _uri: string;
    (sid: string): PoliciesContext;
    get(sid: string): PoliciesContext;
    /**
     * Streams PoliciesInstance records from the API.
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
     * @param { PoliciesListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: PoliciesInstance, done: (err?: Error) => void) => void): void;
    each(params: PoliciesListInstanceEachOptions, callback?: (item: PoliciesInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of PoliciesInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: PoliciesPage) => any): Promise<PoliciesPage>;
    /**
     * Lists PoliciesInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PoliciesListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: PoliciesInstance[]) => any): Promise<PoliciesInstance[]>;
    list(params: PoliciesListInstanceOptions, callback?: (error: Error | null, items: PoliciesInstance[]) => any): Promise<PoliciesInstance[]>;
    /**
     * Retrieve a single page of PoliciesInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PoliciesListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: PoliciesPage) => any): Promise<PoliciesPage>;
    page(params: PoliciesListInstancePageOptions, callback?: (error: Error | null, items: PoliciesPage) => any): Promise<PoliciesPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PoliciesListInstance(version: V1): PoliciesListInstance;
export declare class PoliciesPage extends Page<V1, PoliciesPayload, PoliciesResource, PoliciesInstance> {
    /**
     * Initialize the PoliciesPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: PoliciesSolution);
    /**
     * Build an instance of PoliciesInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: PoliciesResource): PoliciesInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
