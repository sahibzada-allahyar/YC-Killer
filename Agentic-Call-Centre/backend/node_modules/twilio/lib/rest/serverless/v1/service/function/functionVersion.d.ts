/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
import { FunctionVersionContentListInstance } from "./functionVersion/functionVersionContent";
export type FunctionVersionVisibility = "public" | "private" | "protected";
/**
 * Options to pass to each
 */
export interface FunctionVersionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: FunctionVersionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface FunctionVersionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface FunctionVersionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface FunctionVersionContext {
    functionVersionContent: FunctionVersionContentListInstance;
    /**
     * Fetch a FunctionVersionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FunctionVersionInstance
     */
    fetch(callback?: (error: Error | null, item?: FunctionVersionInstance) => any): Promise<FunctionVersionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface FunctionVersionContextSolution {
    serviceSid: string;
    functionSid: string;
    sid: string;
}
export declare class FunctionVersionContextImpl implements FunctionVersionContext {
    protected _version: V1;
    protected _solution: FunctionVersionContextSolution;
    protected _uri: string;
    protected _functionVersionContent?: FunctionVersionContentListInstance;
    constructor(_version: V1, serviceSid: string, functionSid: string, sid: string);
    get functionVersionContent(): FunctionVersionContentListInstance;
    fetch(callback?: (error: Error | null, item?: FunctionVersionInstance) => any): Promise<FunctionVersionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): FunctionVersionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface FunctionVersionPayload extends TwilioResponsePayload {
    function_versions: FunctionVersionResource[];
}
interface FunctionVersionResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    function_sid: string;
    path: string;
    visibility: FunctionVersionVisibility;
    date_created: Date;
    url: string;
    links: Record<string, string>;
}
export declare class FunctionVersionInstance {
    protected _version: V1;
    protected _solution: FunctionVersionContextSolution;
    protected _context?: FunctionVersionContext;
    constructor(_version: V1, payload: FunctionVersionResource, serviceSid: string, functionSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Function Version resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Function Version resource.
     */
    accountSid: string;
    /**
     * The SID of the Service that the Function Version resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the Function resource that is the parent of the Function Version resource.
     */
    functionSid: string;
    /**
     * The URL-friendly string by which the Function Version resource can be referenced. It can be a maximum of 255 characters. All paths begin with a forward slash (\'/\'). If a Function Version creation request is submitted with a path not containing a leading slash, the path will automatically be prepended with one.
     */
    path: string;
    visibility: FunctionVersionVisibility;
    /**
     * The date and time in GMT when the Function Version resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The absolute URL of the Function Version resource.
     */
    url: string;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a FunctionVersionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FunctionVersionInstance
     */
    fetch(callback?: (error: Error | null, item?: FunctionVersionInstance) => any): Promise<FunctionVersionInstance>;
    /**
     * Access the functionVersionContent.
     */
    functionVersionContent(): FunctionVersionContentListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        functionSid: string;
        path: string;
        visibility: FunctionVersionVisibility;
        dateCreated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface FunctionVersionSolution {
    serviceSid: string;
    functionSid: string;
}
export interface FunctionVersionListInstance {
    _version: V1;
    _solution: FunctionVersionSolution;
    _uri: string;
    (sid: string): FunctionVersionContext;
    get(sid: string): FunctionVersionContext;
    /**
     * Streams FunctionVersionInstance records from the API.
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
     * @param { FunctionVersionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: FunctionVersionInstance, done: (err?: Error) => void) => void): void;
    each(params: FunctionVersionListInstanceEachOptions, callback?: (item: FunctionVersionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of FunctionVersionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: FunctionVersionPage) => any): Promise<FunctionVersionPage>;
    /**
     * Lists FunctionVersionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FunctionVersionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: FunctionVersionInstance[]) => any): Promise<FunctionVersionInstance[]>;
    list(params: FunctionVersionListInstanceOptions, callback?: (error: Error | null, items: FunctionVersionInstance[]) => any): Promise<FunctionVersionInstance[]>;
    /**
     * Retrieve a single page of FunctionVersionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FunctionVersionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: FunctionVersionPage) => any): Promise<FunctionVersionPage>;
    page(params: FunctionVersionListInstancePageOptions, callback?: (error: Error | null, items: FunctionVersionPage) => any): Promise<FunctionVersionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function FunctionVersionListInstance(version: V1, serviceSid: string, functionSid: string): FunctionVersionListInstance;
export declare class FunctionVersionPage extends Page<V1, FunctionVersionPayload, FunctionVersionResource, FunctionVersionInstance> {
    /**
     * Initialize the FunctionVersionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: FunctionVersionSolution);
    /**
     * Build an instance of FunctionVersionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: FunctionVersionResource): FunctionVersionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
