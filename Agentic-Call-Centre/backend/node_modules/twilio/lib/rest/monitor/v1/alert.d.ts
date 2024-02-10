/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to each
 */
export interface AlertListInstanceEachOptions {
    /** Only show alerts for this log-level.  Can be: `error`, `warning`, `notice`, or `debug`. */
    logLevel?: string;
    /** Only include alerts that occurred on or after this date and time. Specify the date and time in GMT and format as `YYYY-MM-DD` or `YYYY-MM-DDThh:mm:ssZ`. Queries for alerts older than 30 days are not supported. */
    startDate?: Date;
    /** Only include alerts that occurred on or before this date and time. Specify the date and time in GMT and format as `YYYY-MM-DD` or `YYYY-MM-DDThh:mm:ssZ`. Queries for alerts older than 30 days are not supported. */
    endDate?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AlertInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AlertListInstanceOptions {
    /** Only show alerts for this log-level.  Can be: `error`, `warning`, `notice`, or `debug`. */
    logLevel?: string;
    /** Only include alerts that occurred on or after this date and time. Specify the date and time in GMT and format as `YYYY-MM-DD` or `YYYY-MM-DDThh:mm:ssZ`. Queries for alerts older than 30 days are not supported. */
    startDate?: Date;
    /** Only include alerts that occurred on or before this date and time. Specify the date and time in GMT and format as `YYYY-MM-DD` or `YYYY-MM-DDThh:mm:ssZ`. Queries for alerts older than 30 days are not supported. */
    endDate?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AlertListInstancePageOptions {
    /** Only show alerts for this log-level.  Can be: `error`, `warning`, `notice`, or `debug`. */
    logLevel?: string;
    /** Only include alerts that occurred on or after this date and time. Specify the date and time in GMT and format as `YYYY-MM-DD` or `YYYY-MM-DDThh:mm:ssZ`. Queries for alerts older than 30 days are not supported. */
    startDate?: Date;
    /** Only include alerts that occurred on or before this date and time. Specify the date and time in GMT and format as `YYYY-MM-DD` or `YYYY-MM-DDThh:mm:ssZ`. Queries for alerts older than 30 days are not supported. */
    endDate?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AlertContext {
    /**
     * Fetch a AlertInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AlertInstance
     */
    fetch(callback?: (error: Error | null, item?: AlertInstance) => any): Promise<AlertInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AlertContextSolution {
    sid: string;
}
export declare class AlertContextImpl implements AlertContext {
    protected _version: V1;
    protected _solution: AlertContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: AlertInstance) => any): Promise<AlertInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AlertContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AlertPayload extends TwilioResponsePayload {
    alerts: AlertResource[];
}
interface AlertResource {
    account_sid: string;
    alert_text: string;
    api_version: string;
    date_created: Date;
    date_generated: Date;
    date_updated: Date;
    error_code: string;
    log_level: string;
    more_info: string;
    request_method: string;
    request_url: string;
    request_variables: string;
    resource_sid: string;
    response_body: string;
    response_headers: string;
    sid: string;
    url: string;
    request_headers: string;
    service_sid: string;
}
export declare class AlertInstance {
    protected _version: V1;
    protected _solution: AlertContextSolution;
    protected _context?: AlertContext;
    constructor(_version: V1, payload: AlertResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Alert resource.
     */
    accountSid: string;
    /**
     * The text of the alert.
     */
    alertText: string;
    /**
     * The API version used when the alert was generated.  Can be empty for events that don\'t have a specific API version.
     */
    apiVersion: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the alert was generated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#UTC) format.  Due to buffering, this can be different than `date_created`.
     */
    dateGenerated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The error code for the condition that generated the alert. See the [Error Dictionary](https://www.twilio.com/docs/api/errors) for possible causes and solutions to the error.
     */
    errorCode: string;
    /**
     * The log level.  Can be: `error`, `warning`, `notice`, or `debug`.
     */
    logLevel: string;
    /**
     * The URL of the page in our [Error Dictionary](https://www.twilio.com/docs/api/errors) with more information about the error condition.
     */
    moreInfo: string;
    /**
     * The method used by the request that generated the alert. If the alert was generated by a request we made to your server, this is the method we used. If the alert was generated by a request from your application to our API, this is the method your application used.
     */
    requestMethod: string;
    /**
     * The URL of the request that generated the alert. If the alert was generated by a request we made to your server, this is the URL on your server that generated the alert. If the alert was generated by a request from your application to our API, this is the URL of the resource requested.
     */
    requestUrl: string;
    /**
     * The variables passed in the request that generated the alert. This value is only returned when a single Alert resource is fetched.
     */
    requestVariables: string;
    /**
     * The SID of the resource for which the alert was generated.  For instance, if your server failed to respond to an HTTP request during the flow of a particular call, this value would be the SID of the server.  This value is empty if the alert was not generated for a particular resource.
     */
    resourceSid: string;
    /**
     * The response body of the request that generated the alert. This value is only returned when a single Alert resource is fetched.
     */
    responseBody: string;
    /**
     * The response headers of the request that generated the alert. This value is only returned when a single Alert resource is fetched.
     */
    responseHeaders: string;
    /**
     * The unique string that we created to identify the Alert resource.
     */
    sid: string;
    /**
     * The absolute URL of the Alert resource.
     */
    url: string;
    /**
     * The request headers of the request that generated the alert. This value is only returned when a single Alert resource is fetched.
     */
    requestHeaders: string;
    /**
     * The SID of the service or resource that generated the alert. Can be `null`.
     */
    serviceSid: string;
    private get _proxy();
    /**
     * Fetch a AlertInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AlertInstance
     */
    fetch(callback?: (error: Error | null, item?: AlertInstance) => any): Promise<AlertInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        alertText: string;
        apiVersion: string;
        dateCreated: Date;
        dateGenerated: Date;
        dateUpdated: Date;
        errorCode: string;
        logLevel: string;
        moreInfo: string;
        requestMethod: string;
        requestUrl: string;
        requestVariables: string;
        resourceSid: string;
        responseBody: string;
        responseHeaders: string;
        sid: string;
        url: string;
        requestHeaders: string;
        serviceSid: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AlertSolution {
}
export interface AlertListInstance {
    _version: V1;
    _solution: AlertSolution;
    _uri: string;
    (sid: string): AlertContext;
    get(sid: string): AlertContext;
    /**
     * Streams AlertInstance records from the API.
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
     * @param { AlertListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AlertInstance, done: (err?: Error) => void) => void): void;
    each(params: AlertListInstanceEachOptions, callback?: (item: AlertInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AlertInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AlertPage) => any): Promise<AlertPage>;
    /**
     * Lists AlertInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AlertListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AlertInstance[]) => any): Promise<AlertInstance[]>;
    list(params: AlertListInstanceOptions, callback?: (error: Error | null, items: AlertInstance[]) => any): Promise<AlertInstance[]>;
    /**
     * Retrieve a single page of AlertInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AlertListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AlertPage) => any): Promise<AlertPage>;
    page(params: AlertListInstancePageOptions, callback?: (error: Error | null, items: AlertPage) => any): Promise<AlertPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AlertListInstance(version: V1): AlertListInstance;
export declare class AlertPage extends Page<V1, AlertPayload, AlertResource, AlertInstance> {
    /**
     * Initialize the AlertPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: AlertSolution);
    /**
     * Build an instance of AlertInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AlertResource): AlertInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
