/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2 from "../../../V2";
import { ExecutionStepContextListInstance } from "./executionStep/executionStepContext";
/**
 * Options to pass to each
 */
export interface ExecutionStepListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ExecutionStepInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ExecutionStepListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ExecutionStepListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ExecutionStepContext {
    stepContext: ExecutionStepContextListInstance;
    /**
     * Fetch a ExecutionStepInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExecutionStepInstance
     */
    fetch(callback?: (error: Error | null, item?: ExecutionStepInstance) => any): Promise<ExecutionStepInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ExecutionStepContextSolution {
    flowSid: string;
    executionSid: string;
    sid: string;
}
export declare class ExecutionStepContextImpl implements ExecutionStepContext {
    protected _version: V2;
    protected _solution: ExecutionStepContextSolution;
    protected _uri: string;
    protected _stepContext?: ExecutionStepContextListInstance;
    constructor(_version: V2, flowSid: string, executionSid: string, sid: string);
    get stepContext(): ExecutionStepContextListInstance;
    fetch(callback?: (error: Error | null, item?: ExecutionStepInstance) => any): Promise<ExecutionStepInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ExecutionStepContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ExecutionStepPayload extends TwilioResponsePayload {
    steps: ExecutionStepResource[];
}
interface ExecutionStepResource {
    sid: string;
    account_sid: string;
    flow_sid: string;
    execution_sid: string;
    name: string;
    context: any;
    transitioned_from: string;
    transitioned_to: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class ExecutionStepInstance {
    protected _version: V2;
    protected _solution: ExecutionStepContextSolution;
    protected _context?: ExecutionStepContext;
    constructor(_version: V2, payload: ExecutionStepResource, flowSid: string, executionSid: string, sid?: string);
    /**
     * The unique string that we created to identify the ExecutionStep resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the ExecutionStep resource.
     */
    accountSid: string;
    /**
     * The SID of the Flow.
     */
    flowSid: string;
    /**
     * The SID of the Step\'s Execution resource.
     */
    executionSid: string;
    /**
     * The event that caused the Flow to transition to the Step.
     */
    name: string;
    /**
     * The current state of the Flow\'s Execution. As a flow executes, we save its state in this context. We save data that your widgets can access as variables in configuration fields or in text areas as variable substitution.
     */
    context: any;
    /**
     * The Widget that preceded the Widget for the Step.
     */
    transitionedFrom: string;
    /**
     * The Widget that will follow the Widget for the Step.
     */
    transitionedTo: string;
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
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a ExecutionStepInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExecutionStepInstance
     */
    fetch(callback?: (error: Error | null, item?: ExecutionStepInstance) => any): Promise<ExecutionStepInstance>;
    /**
     * Access the stepContext.
     */
    stepContext(): ExecutionStepContextListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        flowSid: string;
        executionSid: string;
        name: string;
        context: any;
        transitionedFrom: string;
        transitionedTo: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ExecutionStepSolution {
    flowSid: string;
    executionSid: string;
}
export interface ExecutionStepListInstance {
    _version: V2;
    _solution: ExecutionStepSolution;
    _uri: string;
    (sid: string): ExecutionStepContext;
    get(sid: string): ExecutionStepContext;
    /**
     * Streams ExecutionStepInstance records from the API.
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
     * @param { ExecutionStepListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ExecutionStepInstance, done: (err?: Error) => void) => void): void;
    each(params: ExecutionStepListInstanceEachOptions, callback?: (item: ExecutionStepInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ExecutionStepInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ExecutionStepPage) => any): Promise<ExecutionStepPage>;
    /**
     * Lists ExecutionStepInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ExecutionStepListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ExecutionStepInstance[]) => any): Promise<ExecutionStepInstance[]>;
    list(params: ExecutionStepListInstanceOptions, callback?: (error: Error | null, items: ExecutionStepInstance[]) => any): Promise<ExecutionStepInstance[]>;
    /**
     * Retrieve a single page of ExecutionStepInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ExecutionStepListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ExecutionStepPage) => any): Promise<ExecutionStepPage>;
    page(params: ExecutionStepListInstancePageOptions, callback?: (error: Error | null, items: ExecutionStepPage) => any): Promise<ExecutionStepPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ExecutionStepListInstance(version: V2, flowSid: string, executionSid: string): ExecutionStepListInstance;
export declare class ExecutionStepPage extends Page<V2, ExecutionStepPayload, ExecutionStepResource, ExecutionStepInstance> {
    /**
     * Initialize the ExecutionStepPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: ExecutionStepSolution);
    /**
     * Build an instance of ExecutionStepInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ExecutionStepResource): ExecutionStepInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
