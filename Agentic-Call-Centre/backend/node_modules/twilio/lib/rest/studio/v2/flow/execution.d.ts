/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
import { ExecutionContextListInstance } from "./execution/executionContext";
import { ExecutionStepListInstance } from "./execution/executionStep";
export type ExecutionStatus = "active" | "ended";
/**
 * Options to pass to update a ExecutionInstance
 */
export interface ExecutionContextUpdateOptions {
    /**  */
    status: ExecutionStatus;
}
/**
 * Options to pass to create a ExecutionInstance
 */
export interface ExecutionListInstanceCreateOptions {
    /** The Contact phone number to start a Studio Flow Execution, available as variable `{{contact.channel.address}}`. */
    to: string;
    /** The Twilio phone number to send messages or initiate calls from during the Flow\\\'s Execution. Available as variable `{{flow.channel.address}}`. For SMS, this can also be a Messaging Service SID. */
    from: string;
    /** JSON data that will be added to the Flow\\\'s context and that can be accessed as variables inside your Flow. For example, if you pass in `Parameters={\\\"name\\\":\\\"Zeke\\\"}`, a widget in your Flow can reference the variable `{{flow.data.name}}`, which returns \\\"Zeke\\\". Note: the JSON value must explicitly be passed as a string, not as a hash object. Depending on your particular HTTP library, you may need to add quotes or URL encode the JSON string. */
    parameters?: any;
}
/**
 * Options to pass to each
 */
export interface ExecutionListInstanceEachOptions {
    /** Only show Execution resources starting on or after this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time, given as `YYYY-MM-DDThh:mm:ss-hh:mm`. */
    dateCreatedFrom?: Date;
    /** Only show Execution resources starting before this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time, given as `YYYY-MM-DDThh:mm:ss-hh:mm`. */
    dateCreatedTo?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ExecutionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ExecutionListInstanceOptions {
    /** Only show Execution resources starting on or after this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time, given as `YYYY-MM-DDThh:mm:ss-hh:mm`. */
    dateCreatedFrom?: Date;
    /** Only show Execution resources starting before this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time, given as `YYYY-MM-DDThh:mm:ss-hh:mm`. */
    dateCreatedTo?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ExecutionListInstancePageOptions {
    /** Only show Execution resources starting on or after this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time, given as `YYYY-MM-DDThh:mm:ss-hh:mm`. */
    dateCreatedFrom?: Date;
    /** Only show Execution resources starting before this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time, given as `YYYY-MM-DDThh:mm:ss-hh:mm`. */
    dateCreatedTo?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ExecutionContext {
    executionContext: ExecutionContextListInstance;
    steps: ExecutionStepListInstance;
    /**
     * Remove a ExecutionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ExecutionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExecutionInstance
     */
    fetch(callback?: (error: Error | null, item?: ExecutionInstance) => any): Promise<ExecutionInstance>;
    /**
     * Update a ExecutionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExecutionInstance
     */
    update(params: ExecutionContextUpdateOptions, callback?: (error: Error | null, item?: ExecutionInstance) => any): Promise<ExecutionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ExecutionContextSolution {
    flowSid: string;
    sid: string;
}
export declare class ExecutionContextImpl implements ExecutionContext {
    protected _version: V2;
    protected _solution: ExecutionContextSolution;
    protected _uri: string;
    protected _executionContext?: ExecutionContextListInstance;
    protected _steps?: ExecutionStepListInstance;
    constructor(_version: V2, flowSid: string, sid: string);
    get executionContext(): ExecutionContextListInstance;
    get steps(): ExecutionStepListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ExecutionInstance) => any): Promise<ExecutionInstance>;
    update(params: ExecutionContextUpdateOptions, callback?: (error: Error | null, item?: ExecutionInstance) => any): Promise<ExecutionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ExecutionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ExecutionPayload extends TwilioResponsePayload {
    executions: ExecutionResource[];
}
interface ExecutionResource {
    sid: string;
    account_sid: string;
    flow_sid: string;
    contact_channel_address: string;
    context: any;
    status: ExecutionStatus;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class ExecutionInstance {
    protected _version: V2;
    protected _solution: ExecutionContextSolution;
    protected _context?: ExecutionContext;
    constructor(_version: V2, payload: ExecutionResource, flowSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Execution resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Execution resource.
     */
    accountSid: string;
    /**
     * The SID of the Flow.
     */
    flowSid: string;
    /**
     * The phone number, SIP address or Client identifier that triggered the Execution. Phone numbers are in E.164 format (e.g. +16175551212). SIP addresses are formatted as `name@company.com`. Client identifiers are formatted `client:name`.
     */
    contactChannelAddress: string;
    /**
     * The current state of the Flow\'s Execution. As a flow executes, we save its state in this context. We save data that your widgets can access as variables in configuration fields or in text areas as variable substitution.
     */
    context: any;
    status: ExecutionStatus;
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
     * The URLs of nested resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a ExecutionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ExecutionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExecutionInstance
     */
    fetch(callback?: (error: Error | null, item?: ExecutionInstance) => any): Promise<ExecutionInstance>;
    /**
     * Update a ExecutionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExecutionInstance
     */
    update(params: ExecutionContextUpdateOptions, callback?: (error: Error | null, item?: ExecutionInstance) => any): Promise<ExecutionInstance>;
    /**
     * Access the executionContext.
     */
    executionContext(): ExecutionContextListInstance;
    /**
     * Access the steps.
     */
    steps(): ExecutionStepListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        flowSid: string;
        contactChannelAddress: string;
        context: any;
        status: ExecutionStatus;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ExecutionSolution {
    flowSid: string;
}
export interface ExecutionListInstance {
    _version: V2;
    _solution: ExecutionSolution;
    _uri: string;
    (sid: string): ExecutionContext;
    get(sid: string): ExecutionContext;
    /**
     * Create a ExecutionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExecutionInstance
     */
    create(params: ExecutionListInstanceCreateOptions, callback?: (error: Error | null, item?: ExecutionInstance) => any): Promise<ExecutionInstance>;
    /**
     * Streams ExecutionInstance records from the API.
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
     * @param { ExecutionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ExecutionInstance, done: (err?: Error) => void) => void): void;
    each(params: ExecutionListInstanceEachOptions, callback?: (item: ExecutionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ExecutionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ExecutionPage) => any): Promise<ExecutionPage>;
    /**
     * Lists ExecutionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ExecutionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ExecutionInstance[]) => any): Promise<ExecutionInstance[]>;
    list(params: ExecutionListInstanceOptions, callback?: (error: Error | null, items: ExecutionInstance[]) => any): Promise<ExecutionInstance[]>;
    /**
     * Retrieve a single page of ExecutionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ExecutionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ExecutionPage) => any): Promise<ExecutionPage>;
    page(params: ExecutionListInstancePageOptions, callback?: (error: Error | null, items: ExecutionPage) => any): Promise<ExecutionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ExecutionListInstance(version: V2, flowSid: string): ExecutionListInstance;
export declare class ExecutionPage extends Page<V2, ExecutionPayload, ExecutionResource, ExecutionInstance> {
    /**
     * Initialize the ExecutionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: ExecutionSolution);
    /**
     * Build an instance of ExecutionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ExecutionResource): ExecutionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
