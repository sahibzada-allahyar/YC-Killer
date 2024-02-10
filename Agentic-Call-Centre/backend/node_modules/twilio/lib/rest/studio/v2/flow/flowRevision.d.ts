/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
export type FlowRevisionStatus = "draft" | "published";
/**
 * Options to pass to each
 */
export interface FlowRevisionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: FlowRevisionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface FlowRevisionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface FlowRevisionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface FlowRevisionContext {
    /**
     * Fetch a FlowRevisionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlowRevisionInstance
     */
    fetch(callback?: (error: Error | null, item?: FlowRevisionInstance) => any): Promise<FlowRevisionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface FlowRevisionContextSolution {
    sid: string;
    revision: string;
}
export declare class FlowRevisionContextImpl implements FlowRevisionContext {
    protected _version: V2;
    protected _solution: FlowRevisionContextSolution;
    protected _uri: string;
    constructor(_version: V2, sid: string, revision: string);
    fetch(callback?: (error: Error | null, item?: FlowRevisionInstance) => any): Promise<FlowRevisionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): FlowRevisionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface FlowRevisionPayload extends TwilioResponsePayload {
    revisions: FlowRevisionResource[];
}
interface FlowRevisionResource {
    sid: string;
    account_sid: string;
    friendly_name: string;
    definition: any;
    status: FlowRevisionStatus;
    revision: number;
    commit_message: string;
    valid: boolean;
    errors: Array<any>;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class FlowRevisionInstance {
    protected _version: V2;
    protected _solution: FlowRevisionContextSolution;
    protected _context?: FlowRevisionContext;
    constructor(_version: V2, payload: FlowRevisionResource, sid: string, revision?: string);
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
    /**
     * JSON representation of flow definition.
     */
    definition: any;
    status: FlowRevisionStatus;
    /**
     * The latest revision number of the Flow\'s definition.
     */
    revision: number;
    /**
     * Description of change made in the revision.
     */
    commitMessage: string;
    /**
     * Boolean if the flow definition is valid.
     */
    valid: boolean;
    /**
     * List of error in the flow definition.
     */
    errors: Array<any>;
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
    private get _proxy();
    /**
     * Fetch a FlowRevisionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlowRevisionInstance
     */
    fetch(callback?: (error: Error | null, item?: FlowRevisionInstance) => any): Promise<FlowRevisionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        friendlyName: string;
        definition: any;
        status: FlowRevisionStatus;
        revision: number;
        commitMessage: string;
        valid: boolean;
        errors: any[];
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface FlowRevisionSolution {
    sid: string;
}
export interface FlowRevisionListInstance {
    _version: V2;
    _solution: FlowRevisionSolution;
    _uri: string;
    (revision: string): FlowRevisionContext;
    get(revision: string): FlowRevisionContext;
    /**
     * Streams FlowRevisionInstance records from the API.
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
     * @param { FlowRevisionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: FlowRevisionInstance, done: (err?: Error) => void) => void): void;
    each(params: FlowRevisionListInstanceEachOptions, callback?: (item: FlowRevisionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of FlowRevisionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: FlowRevisionPage) => any): Promise<FlowRevisionPage>;
    /**
     * Lists FlowRevisionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FlowRevisionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: FlowRevisionInstance[]) => any): Promise<FlowRevisionInstance[]>;
    list(params: FlowRevisionListInstanceOptions, callback?: (error: Error | null, items: FlowRevisionInstance[]) => any): Promise<FlowRevisionInstance[]>;
    /**
     * Retrieve a single page of FlowRevisionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FlowRevisionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: FlowRevisionPage) => any): Promise<FlowRevisionPage>;
    page(params: FlowRevisionListInstancePageOptions, callback?: (error: Error | null, items: FlowRevisionPage) => any): Promise<FlowRevisionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function FlowRevisionListInstance(version: V2, sid: string): FlowRevisionListInstance;
export declare class FlowRevisionPage extends Page<V2, FlowRevisionPayload, FlowRevisionResource, FlowRevisionInstance> {
    /**
     * Initialize the FlowRevisionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: FlowRevisionSolution);
    /**
     * Build an instance of FlowRevisionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: FlowRevisionResource): FlowRevisionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
