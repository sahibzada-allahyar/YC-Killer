/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
import { StepContextListInstance } from "./step/stepContext";
/**
 * Options to pass to each
 */
export interface StepListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: StepInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface StepListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface StepListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface StepContext {
    stepContext: StepContextListInstance;
    /**
     * Fetch a StepInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed StepInstance
     */
    fetch(callback?: (error: Error | null, item?: StepInstance) => any): Promise<StepInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface StepContextSolution {
    flowSid: string;
    engagementSid: string;
    sid: string;
}
export declare class StepContextImpl implements StepContext {
    protected _version: V1;
    protected _solution: StepContextSolution;
    protected _uri: string;
    protected _stepContext?: StepContextListInstance;
    constructor(_version: V1, flowSid: string, engagementSid: string, sid: string);
    get stepContext(): StepContextListInstance;
    fetch(callback?: (error: Error | null, item?: StepInstance) => any): Promise<StepInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): StepContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface StepPayload extends TwilioResponsePayload {
    steps: StepResource[];
}
interface StepResource {
    sid: string;
    account_sid: string;
    flow_sid: string;
    engagement_sid: string;
    name: string;
    context: any;
    transitioned_from: string;
    transitioned_to: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class StepInstance {
    protected _version: V1;
    protected _solution: StepContextSolution;
    protected _context?: StepContext;
    constructor(_version: V1, payload: StepResource, flowSid: string, engagementSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Step resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Step resource.
     */
    accountSid: string;
    /**
     * The SID of the Flow.
     */
    flowSid: string;
    /**
     * The SID of the Engagement.
     */
    engagementSid: string;
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
     * Fetch a StepInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed StepInstance
     */
    fetch(callback?: (error: Error | null, item?: StepInstance) => any): Promise<StepInstance>;
    /**
     * Access the stepContext.
     */
    stepContext(): StepContextListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        flowSid: string;
        engagementSid: string;
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
export interface StepSolution {
    flowSid: string;
    engagementSid: string;
}
export interface StepListInstance {
    _version: V1;
    _solution: StepSolution;
    _uri: string;
    (sid: string): StepContext;
    get(sid: string): StepContext;
    /**
     * Streams StepInstance records from the API.
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
     * @param { StepListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: StepInstance, done: (err?: Error) => void) => void): void;
    each(params: StepListInstanceEachOptions, callback?: (item: StepInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of StepInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: StepPage) => any): Promise<StepPage>;
    /**
     * Lists StepInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { StepListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: StepInstance[]) => any): Promise<StepInstance[]>;
    list(params: StepListInstanceOptions, callback?: (error: Error | null, items: StepInstance[]) => any): Promise<StepInstance[]>;
    /**
     * Retrieve a single page of StepInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { StepListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: StepPage) => any): Promise<StepPage>;
    page(params: StepListInstancePageOptions, callback?: (error: Error | null, items: StepPage) => any): Promise<StepPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function StepListInstance(version: V1, flowSid: string, engagementSid: string): StepListInstance;
export declare class StepPage extends Page<V1, StepPayload, StepResource, StepInstance> {
    /**
     * Initialize the StepPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: StepSolution);
    /**
     * Build an instance of StepInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: StepResource): StepInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
