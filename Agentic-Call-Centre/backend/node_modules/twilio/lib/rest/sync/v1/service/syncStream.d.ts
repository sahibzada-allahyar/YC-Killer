/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { StreamMessageListInstance } from "./syncStream/streamMessage";
/**
 * Options to pass to update a SyncStreamInstance
 */
export interface SyncStreamContextUpdateOptions {
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the Stream expires and is deleted (time-to-live). */
    ttl?: number;
}
/**
 * Options to pass to create a SyncStreamInstance
 */
export interface SyncStreamListInstanceCreateOptions {
    /** An application-defined string that uniquely identifies the resource. This value must be unique within its Service and it can be up to 320 characters long. The `unique_name` value can be used as an alternative to the `sid` in the URL path to address the resource. */
    uniqueName?: string;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the Stream expires and is deleted (time-to-live). */
    ttl?: number;
}
/**
 * Options to pass to each
 */
export interface SyncStreamListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SyncStreamInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SyncStreamListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SyncStreamListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SyncStreamContext {
    streamMessages: StreamMessageListInstance;
    /**
     * Remove a SyncStreamInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SyncStreamInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncStreamInstance
     */
    fetch(callback?: (error: Error | null, item?: SyncStreamInstance) => any): Promise<SyncStreamInstance>;
    /**
     * Update a SyncStreamInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncStreamInstance
     */
    update(callback?: (error: Error | null, item?: SyncStreamInstance) => any): Promise<SyncStreamInstance>;
    /**
     * Update a SyncStreamInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncStreamInstance
     */
    update(params: SyncStreamContextUpdateOptions, callback?: (error: Error | null, item?: SyncStreamInstance) => any): Promise<SyncStreamInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SyncStreamContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class SyncStreamContextImpl implements SyncStreamContext {
    protected _version: V1;
    protected _solution: SyncStreamContextSolution;
    protected _uri: string;
    protected _streamMessages?: StreamMessageListInstance;
    constructor(_version: V1, serviceSid: string, sid: string);
    get streamMessages(): StreamMessageListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SyncStreamInstance) => any): Promise<SyncStreamInstance>;
    update(params?: SyncStreamContextUpdateOptions | ((error: Error | null, item?: SyncStreamInstance) => any), callback?: (error: Error | null, item?: SyncStreamInstance) => any): Promise<SyncStreamInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SyncStreamContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SyncStreamPayload extends TwilioResponsePayload {
    streams: SyncStreamResource[];
}
interface SyncStreamResource {
    sid: string;
    unique_name: string;
    account_sid: string;
    service_sid: string;
    url: string;
    links: Record<string, string>;
    date_expires: Date;
    date_created: Date;
    date_updated: Date;
    created_by: string;
}
export declare class SyncStreamInstance {
    protected _version: V1;
    protected _solution: SyncStreamContextSolution;
    protected _context?: SyncStreamContext;
    constructor(_version: V1, payload: SyncStreamResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Sync Stream resource.
     */
    sid: string;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used in place of the resource\'s `sid` in the URL to address the resource.
     */
    uniqueName: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Sync Stream resource.
     */
    accountSid: string;
    /**
     * The SID of the [Sync Service](https://www.twilio.com/docs/sync/api/service) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The absolute URL of the Message Stream resource.
     */
    url: string;
    /**
     * The URLs of the Stream\'s nested resources.
     */
    links: Record<string, string>;
    /**
     * The date and time in GMT when the Message Stream expires and will be deleted, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. If the Message Stream does not expire, this value is `null`. The Stream might not be deleted immediately after it expires.
     */
    dateExpires: Date;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The identity of the Stream\'s creator. If the Stream is created from the client SDK, the value matches the Access Token\'s `identity` field. If the Stream was created from the REST API, the value is \'system\'.
     */
    createdBy: string;
    private get _proxy();
    /**
     * Remove a SyncStreamInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SyncStreamInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncStreamInstance
     */
    fetch(callback?: (error: Error | null, item?: SyncStreamInstance) => any): Promise<SyncStreamInstance>;
    /**
     * Update a SyncStreamInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncStreamInstance
     */
    update(callback?: (error: Error | null, item?: SyncStreamInstance) => any): Promise<SyncStreamInstance>;
    /**
     * Update a SyncStreamInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncStreamInstance
     */
    update(params: SyncStreamContextUpdateOptions, callback?: (error: Error | null, item?: SyncStreamInstance) => any): Promise<SyncStreamInstance>;
    /**
     * Access the streamMessages.
     */
    streamMessages(): StreamMessageListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        uniqueName: string;
        accountSid: string;
        serviceSid: string;
        url: string;
        links: Record<string, string>;
        dateExpires: Date;
        dateCreated: Date;
        dateUpdated: Date;
        createdBy: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SyncStreamSolution {
    serviceSid: string;
}
export interface SyncStreamListInstance {
    _version: V1;
    _solution: SyncStreamSolution;
    _uri: string;
    (sid: string): SyncStreamContext;
    get(sid: string): SyncStreamContext;
    /**
     * Create a SyncStreamInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncStreamInstance
     */
    create(callback?: (error: Error | null, item?: SyncStreamInstance) => any): Promise<SyncStreamInstance>;
    /**
     * Create a SyncStreamInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncStreamInstance
     */
    create(params: SyncStreamListInstanceCreateOptions, callback?: (error: Error | null, item?: SyncStreamInstance) => any): Promise<SyncStreamInstance>;
    /**
     * Streams SyncStreamInstance records from the API.
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
     * @param { SyncStreamListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SyncStreamInstance, done: (err?: Error) => void) => void): void;
    each(params: SyncStreamListInstanceEachOptions, callback?: (item: SyncStreamInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SyncStreamInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SyncStreamPage) => any): Promise<SyncStreamPage>;
    /**
     * Lists SyncStreamInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SyncStreamListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SyncStreamInstance[]) => any): Promise<SyncStreamInstance[]>;
    list(params: SyncStreamListInstanceOptions, callback?: (error: Error | null, items: SyncStreamInstance[]) => any): Promise<SyncStreamInstance[]>;
    /**
     * Retrieve a single page of SyncStreamInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SyncStreamListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SyncStreamPage) => any): Promise<SyncStreamPage>;
    page(params: SyncStreamListInstancePageOptions, callback?: (error: Error | null, items: SyncStreamPage) => any): Promise<SyncStreamPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SyncStreamListInstance(version: V1, serviceSid: string): SyncStreamListInstance;
export declare class SyncStreamPage extends Page<V1, SyncStreamPayload, SyncStreamResource, SyncStreamInstance> {
    /**
     * Initialize the SyncStreamPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SyncStreamSolution);
    /**
     * Build an instance of SyncStreamInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SyncStreamResource): SyncStreamInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
