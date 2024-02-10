/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
export type LogLevel = "info" | "warn" | "error";
/**
 * Options to pass to each
 */
export interface LogListInstanceEachOptions {
    /** The SID of the function whose invocation produced the Log resources to read. */
    functionSid?: string;
    /** The date/time (in GMT, ISO 8601) after which the Log resources must have been created. Defaults to 1 day prior to current date/time. */
    startDate?: Date;
    /** The date/time (in GMT, ISO 8601) before which the Log resources must have been created. Defaults to current date/time. */
    endDate?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: LogInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface LogListInstanceOptions {
    /** The SID of the function whose invocation produced the Log resources to read. */
    functionSid?: string;
    /** The date/time (in GMT, ISO 8601) after which the Log resources must have been created. Defaults to 1 day prior to current date/time. */
    startDate?: Date;
    /** The date/time (in GMT, ISO 8601) before which the Log resources must have been created. Defaults to current date/time. */
    endDate?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface LogListInstancePageOptions {
    /** The SID of the function whose invocation produced the Log resources to read. */
    functionSid?: string;
    /** The date/time (in GMT, ISO 8601) after which the Log resources must have been created. Defaults to 1 day prior to current date/time. */
    startDate?: Date;
    /** The date/time (in GMT, ISO 8601) before which the Log resources must have been created. Defaults to current date/time. */
    endDate?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface LogContext {
    /**
     * Fetch a LogInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed LogInstance
     */
    fetch(callback?: (error: Error | null, item?: LogInstance) => any): Promise<LogInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface LogContextSolution {
    serviceSid: string;
    environmentSid: string;
    sid: string;
}
export declare class LogContextImpl implements LogContext {
    protected _version: V1;
    protected _solution: LogContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, environmentSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: LogInstance) => any): Promise<LogInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): LogContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface LogPayload extends TwilioResponsePayload {
    logs: LogResource[];
}
interface LogResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    environment_sid: string;
    build_sid: string;
    deployment_sid: string;
    function_sid: string;
    request_sid: string;
    level: LogLevel;
    message: string;
    date_created: Date;
    url: string;
}
export declare class LogInstance {
    protected _version: V1;
    protected _solution: LogContextSolution;
    protected _context?: LogContext;
    constructor(_version: V1, payload: LogResource, serviceSid: string, environmentSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Log resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Log resource.
     */
    accountSid: string;
    /**
     * The SID of the Service that the Log resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the environment in which the log occurred.
     */
    environmentSid: string;
    /**
     * The SID of the build that corresponds to the log.
     */
    buildSid: string;
    /**
     * The SID of the deployment that corresponds to the log.
     */
    deploymentSid: string;
    /**
     * The SID of the function whose invocation produced the log.
     */
    functionSid: string;
    /**
     * The SID of the request associated with the log.
     */
    requestSid: string;
    level: LogLevel;
    /**
     * The log message.
     */
    message: string;
    /**
     * The date and time in GMT when the Log resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The absolute URL of the Log resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a LogInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed LogInstance
     */
    fetch(callback?: (error: Error | null, item?: LogInstance) => any): Promise<LogInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        environmentSid: string;
        buildSid: string;
        deploymentSid: string;
        functionSid: string;
        requestSid: string;
        level: LogLevel;
        message: string;
        dateCreated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface LogSolution {
    serviceSid: string;
    environmentSid: string;
}
export interface LogListInstance {
    _version: V1;
    _solution: LogSolution;
    _uri: string;
    (sid: string): LogContext;
    get(sid: string): LogContext;
    /**
     * Streams LogInstance records from the API.
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
     * @param { LogListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: LogInstance, done: (err?: Error) => void) => void): void;
    each(params: LogListInstanceEachOptions, callback?: (item: LogInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of LogInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: LogPage) => any): Promise<LogPage>;
    /**
     * Lists LogInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { LogListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: LogInstance[]) => any): Promise<LogInstance[]>;
    list(params: LogListInstanceOptions, callback?: (error: Error | null, items: LogInstance[]) => any): Promise<LogInstance[]>;
    /**
     * Retrieve a single page of LogInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { LogListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: LogPage) => any): Promise<LogPage>;
    page(params: LogListInstancePageOptions, callback?: (error: Error | null, items: LogPage) => any): Promise<LogPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function LogListInstance(version: V1, serviceSid: string, environmentSid: string): LogListInstance;
export declare class LogPage extends Page<V1, LogPayload, LogResource, LogInstance> {
    /**
     * Initialize the LogPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: LogSolution);
    /**
     * Build an instance of LogInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: LogResource): LogInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
