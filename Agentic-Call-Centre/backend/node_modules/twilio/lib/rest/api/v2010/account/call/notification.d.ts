/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2010 from "../../../V2010";
/**
 * Options to pass to each
 */
export interface NotificationListInstanceEachOptions {
    /** Only read notifications of the specified log level. Can be:  `0` to read only ERROR notifications or `1` to read only WARNING notifications. By default, all notifications are read. */
    log?: number;
    /** Only show notifications for the specified date, formatted as `YYYY-MM-DD`. You can also specify an inequality, such as `<=YYYY-MM-DD` for messages logged at or before midnight on a date, or `>=YYYY-MM-DD` for messages logged at or after midnight on a date. */
    messageDate?: Date;
    /** Only show notifications for the specified date, formatted as `YYYY-MM-DD`. You can also specify an inequality, such as `<=YYYY-MM-DD` for messages logged at or before midnight on a date, or `>=YYYY-MM-DD` for messages logged at or after midnight on a date. */
    messageDateBefore?: Date;
    /** Only show notifications for the specified date, formatted as `YYYY-MM-DD`. You can also specify an inequality, such as `<=YYYY-MM-DD` for messages logged at or before midnight on a date, or `>=YYYY-MM-DD` for messages logged at or after midnight on a date. */
    messageDateAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: NotificationInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface NotificationListInstanceOptions {
    /** Only read notifications of the specified log level. Can be:  `0` to read only ERROR notifications or `1` to read only WARNING notifications. By default, all notifications are read. */
    log?: number;
    /** Only show notifications for the specified date, formatted as `YYYY-MM-DD`. You can also specify an inequality, such as `<=YYYY-MM-DD` for messages logged at or before midnight on a date, or `>=YYYY-MM-DD` for messages logged at or after midnight on a date. */
    messageDate?: Date;
    /** Only show notifications for the specified date, formatted as `YYYY-MM-DD`. You can also specify an inequality, such as `<=YYYY-MM-DD` for messages logged at or before midnight on a date, or `>=YYYY-MM-DD` for messages logged at or after midnight on a date. */
    messageDateBefore?: Date;
    /** Only show notifications for the specified date, formatted as `YYYY-MM-DD`. You can also specify an inequality, such as `<=YYYY-MM-DD` for messages logged at or before midnight on a date, or `>=YYYY-MM-DD` for messages logged at or after midnight on a date. */
    messageDateAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface NotificationListInstancePageOptions {
    /** Only read notifications of the specified log level. Can be:  `0` to read only ERROR notifications or `1` to read only WARNING notifications. By default, all notifications are read. */
    log?: number;
    /** Only show notifications for the specified date, formatted as `YYYY-MM-DD`. You can also specify an inequality, such as `<=YYYY-MM-DD` for messages logged at or before midnight on a date, or `>=YYYY-MM-DD` for messages logged at or after midnight on a date. */
    messageDate?: Date;
    /** Only show notifications for the specified date, formatted as `YYYY-MM-DD`. You can also specify an inequality, such as `<=YYYY-MM-DD` for messages logged at or before midnight on a date, or `>=YYYY-MM-DD` for messages logged at or after midnight on a date. */
    messageDateBefore?: Date;
    /** Only show notifications for the specified date, formatted as `YYYY-MM-DD`. You can also specify an inequality, such as `<=YYYY-MM-DD` for messages logged at or before midnight on a date, or `>=YYYY-MM-DD` for messages logged at or after midnight on a date. */
    messageDateAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface NotificationContext {
    /**
     * Fetch a NotificationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NotificationInstance
     */
    fetch(callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface NotificationContextSolution {
    accountSid: string;
    callSid: string;
    sid: string;
}
export declare class NotificationContextImpl implements NotificationContext {
    protected _version: V2010;
    protected _solution: NotificationContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, callSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): NotificationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface NotificationPayload extends TwilioResponsePayload {
    notifications: NotificationResource[];
}
interface NotificationResource {
    account_sid: string;
    api_version: string;
    call_sid: string;
    date_created: Date;
    date_updated: Date;
    error_code: string;
    log: string;
    message_date: Date;
    message_text: string;
    more_info: string;
    request_method: string;
    request_url: string;
    request_variables: string;
    response_body: string;
    response_headers: string;
    sid: string;
    uri: string;
}
export declare class NotificationInstance {
    protected _version: V2010;
    protected _solution: NotificationContextSolution;
    protected _context?: NotificationContext;
    constructor(_version: V2010, payload: NotificationResource, accountSid: string, callSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Call Notification resource.
     */
    accountSid: string;
    /**
     * The API version used to create the Call Notification resource.
     */
    apiVersion: string;
    /**
     * The SID of the [Call](https://www.twilio.com/docs/voice/api/call-resource) the Call Notification resource is associated with.
     */
    callSid: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * A unique error code for the error condition that is described in our [Error Dictionary](https://www.twilio.com/docs/api/errors).
     */
    errorCode: string;
    /**
     * An integer log level that corresponds to the type of notification: `0` is ERROR, `1` is WARNING.
     */
    log: string;
    /**
     * The date the notification was actually generated in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format. Message buffering can cause this value to differ from `date_created`.
     */
    messageDate: Date;
    /**
     * The text of the notification.
     */
    messageText: string;
    /**
     * The URL for more information about the error condition. This value is a page in our [Error Dictionary](https://www.twilio.com/docs/api/errors).
     */
    moreInfo: string;
    /**
     * The HTTP method used to generate the notification. If the notification was generated during a phone call, this is the HTTP Method used to request the resource on your server. If the notification was generated by your use of our REST API, this is the HTTP method used to call the resource on our servers.
     */
    requestMethod: string;
    /**
     * The URL of the resource that generated the notification. If the notification was generated during a phone call, this is the URL of the resource on your server that caused the notification. If the notification was generated by your use of our REST API, this is the URL of the resource you called.
     */
    requestUrl: string;
    /**
     * The HTTP GET or POST variables we sent to your server. However, if the notification was generated by our REST API, this contains the HTTP POST or PUT variables you sent to our API.
     */
    requestVariables: string;
    /**
     * The HTTP body returned by your server.
     */
    responseBody: string;
    /**
     * The HTTP headers returned by your server.
     */
    responseHeaders: string;
    /**
     * The unique string that that we created to identify the Call Notification resource.
     */
    sid: string;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Fetch a NotificationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NotificationInstance
     */
    fetch(callback?: (error: Error | null, item?: NotificationInstance) => any): Promise<NotificationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        apiVersion: string;
        callSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        errorCode: string;
        log: string;
        messageDate: Date;
        messageText: string;
        moreInfo: string;
        requestMethod: string;
        requestUrl: string;
        requestVariables: string;
        responseBody: string;
        responseHeaders: string;
        sid: string;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface NotificationSolution {
    accountSid: string;
    callSid: string;
}
export interface NotificationListInstance {
    _version: V2010;
    _solution: NotificationSolution;
    _uri: string;
    (sid: string): NotificationContext;
    get(sid: string): NotificationContext;
    /**
     * Streams NotificationInstance records from the API.
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
     * @param { NotificationListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: NotificationInstance, done: (err?: Error) => void) => void): void;
    each(params: NotificationListInstanceEachOptions, callback?: (item: NotificationInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of NotificationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: NotificationPage) => any): Promise<NotificationPage>;
    /**
     * Lists NotificationInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { NotificationListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: NotificationInstance[]) => any): Promise<NotificationInstance[]>;
    list(params: NotificationListInstanceOptions, callback?: (error: Error | null, items: NotificationInstance[]) => any): Promise<NotificationInstance[]>;
    /**
     * Retrieve a single page of NotificationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { NotificationListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: NotificationPage) => any): Promise<NotificationPage>;
    page(params: NotificationListInstancePageOptions, callback?: (error: Error | null, items: NotificationPage) => any): Promise<NotificationPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function NotificationListInstance(version: V2010, accountSid: string, callSid: string): NotificationListInstance;
export declare class NotificationPage extends Page<V2010, NotificationPayload, NotificationResource, NotificationInstance> {
    /**
     * Initialize the NotificationPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: NotificationSolution);
    /**
     * Build an instance of NotificationInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: NotificationResource): NotificationInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
