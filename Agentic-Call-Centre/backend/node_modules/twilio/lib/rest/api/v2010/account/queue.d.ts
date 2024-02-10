/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
import { MemberListInstance } from "./queue/member";
/**
 * Options to pass to update a QueueInstance
 */
export interface QueueContextUpdateOptions {
    /** A descriptive string that you created to describe this resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** The maximum number of calls allowed to be in the queue. The default is 1000. The maximum is 5000. */
    maxSize?: number;
}
/**
 * Options to pass to create a QueueInstance
 */
export interface QueueListInstanceCreateOptions {
    /** A descriptive string that you created to describe this resource. It can be up to 64 characters long. */
    friendlyName: string;
    /** The maximum number of calls allowed to be in the queue. The default is 1000. The maximum is 5000. */
    maxSize?: number;
}
/**
 * Options to pass to each
 */
export interface QueueListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: QueueInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface QueueListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface QueueListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface QueueContext {
    members: MemberListInstance;
    /**
     * Remove a QueueInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a QueueInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed QueueInstance
     */
    fetch(callback?: (error: Error | null, item?: QueueInstance) => any): Promise<QueueInstance>;
    /**
     * Update a QueueInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed QueueInstance
     */
    update(callback?: (error: Error | null, item?: QueueInstance) => any): Promise<QueueInstance>;
    /**
     * Update a QueueInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed QueueInstance
     */
    update(params: QueueContextUpdateOptions, callback?: (error: Error | null, item?: QueueInstance) => any): Promise<QueueInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface QueueContextSolution {
    accountSid: string;
    sid: string;
}
export declare class QueueContextImpl implements QueueContext {
    protected _version: V2010;
    protected _solution: QueueContextSolution;
    protected _uri: string;
    protected _members?: MemberListInstance;
    constructor(_version: V2010, accountSid: string, sid: string);
    get members(): MemberListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: QueueInstance) => any): Promise<QueueInstance>;
    update(params?: QueueContextUpdateOptions | ((error: Error | null, item?: QueueInstance) => any), callback?: (error: Error | null, item?: QueueInstance) => any): Promise<QueueInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): QueueContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface QueuePayload extends TwilioResponsePayload {
    queues: QueueResource[];
}
interface QueueResource {
    date_updated: Date;
    current_size: number;
    friendly_name: string;
    uri: string;
    account_sid: string;
    average_wait_time: number;
    sid: string;
    date_created: Date;
    max_size: number;
}
export declare class QueueInstance {
    protected _version: V2010;
    protected _solution: QueueContextSolution;
    protected _context?: QueueContext;
    constructor(_version: V2010, payload: QueueResource, accountSid: string, sid?: string);
    /**
     * The date and time in GMT that this resource was last updated, specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The number of calls currently in the queue.
     */
    currentSize: number;
    /**
     * A string that you assigned to describe this resource.
     */
    friendlyName: string;
    /**
     * The URI of this resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created this Queue resource.
     */
    accountSid: string;
    /**
     *  The average wait time in seconds of the members in this queue. This is calculated at the time of the request.
     */
    averageWaitTime: number;
    /**
     * The unique string that that we created to identify this Queue resource.
     */
    sid: string;
    /**
     * The date and time in GMT that this resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     *  The maximum number of calls that can be in the queue. The default is 1000 and the maximum is 5000.
     */
    maxSize: number;
    private get _proxy();
    /**
     * Remove a QueueInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a QueueInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed QueueInstance
     */
    fetch(callback?: (error: Error | null, item?: QueueInstance) => any): Promise<QueueInstance>;
    /**
     * Update a QueueInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed QueueInstance
     */
    update(callback?: (error: Error | null, item?: QueueInstance) => any): Promise<QueueInstance>;
    /**
     * Update a QueueInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed QueueInstance
     */
    update(params: QueueContextUpdateOptions, callback?: (error: Error | null, item?: QueueInstance) => any): Promise<QueueInstance>;
    /**
     * Access the members.
     */
    members(): MemberListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        dateUpdated: Date;
        currentSize: number;
        friendlyName: string;
        uri: string;
        accountSid: string;
        averageWaitTime: number;
        sid: string;
        dateCreated: Date;
        maxSize: number;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface QueueSolution {
    accountSid: string;
}
export interface QueueListInstance {
    _version: V2010;
    _solution: QueueSolution;
    _uri: string;
    (sid: string): QueueContext;
    get(sid: string): QueueContext;
    /**
     * Create a QueueInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed QueueInstance
     */
    create(params: QueueListInstanceCreateOptions, callback?: (error: Error | null, item?: QueueInstance) => any): Promise<QueueInstance>;
    /**
     * Streams QueueInstance records from the API.
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
     * @param { QueueListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: QueueInstance, done: (err?: Error) => void) => void): void;
    each(params: QueueListInstanceEachOptions, callback?: (item: QueueInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of QueueInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: QueuePage) => any): Promise<QueuePage>;
    /**
     * Lists QueueInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { QueueListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: QueueInstance[]) => any): Promise<QueueInstance[]>;
    list(params: QueueListInstanceOptions, callback?: (error: Error | null, items: QueueInstance[]) => any): Promise<QueueInstance[]>;
    /**
     * Retrieve a single page of QueueInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { QueueListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: QueuePage) => any): Promise<QueuePage>;
    page(params: QueueListInstancePageOptions, callback?: (error: Error | null, items: QueuePage) => any): Promise<QueuePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function QueueListInstance(version: V2010, accountSid: string): QueueListInstance;
export declare class QueuePage extends Page<V2010, QueuePayload, QueueResource, QueueInstance> {
    /**
     * Initialize the QueuePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: QueueSolution);
    /**
     * Build an instance of QueueInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: QueueResource): QueueInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
