/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type MediaProcessorOrder = "asc" | "desc";
export type MediaProcessorStatus = "failed" | "started" | "ended";
export type MediaProcessorUpdateStatus = "ended";
/**
 * Options to pass to update a MediaProcessorInstance
 */
export interface MediaProcessorContextUpdateOptions {
    /**  */
    status: MediaProcessorUpdateStatus;
}
/**
 * Options to pass to create a MediaProcessorInstance
 */
export interface MediaProcessorListInstanceCreateOptions {
    /** The [Media Extension](/docs/live/media-extensions-overview) name or URL. Ex: `video-composer-v2` */
    extension: string;
    /** The context of the Media Extension, represented as a JSON dictionary. See the documentation for the specific [Media Extension](/docs/live/media-extensions-overview) you are using for more information about the context to send. */
    extensionContext: string;
    /** User-defined environment variables for the Media Extension, represented as a JSON dictionary of key/value strings. See the documentation for the specific [Media Extension](/docs/live/media-extensions-overview) you are using for more information about whether you need to provide this. */
    extensionEnvironment?: any;
    /** The URL to which Twilio will send asynchronous webhook requests for every MediaProcessor event. See [Status Callbacks](/docs/live/api/status-callbacks) for details. */
    statusCallback?: string;
    /** The HTTP method Twilio should use to call the `status_callback` URL. Can be `POST` or `GET` and the default is `POST`. */
    statusCallbackMethod?: string;
    /** The maximum time, in seconds, that the MediaProcessor can run before automatically ends. The default value is 300 seconds, and the maximum value is 90000 seconds. Once this maximum duration is reached, Twilio will end the MediaProcessor, regardless of whether media is still streaming. */
    maxDuration?: number;
}
/**
 * Options to pass to each
 */
export interface MediaProcessorListInstanceEachOptions {
    /** The sort order of the list by `date_created`. Can be: `asc` (ascending) or `desc` (descending) with `desc` as the default. */
    order?: MediaProcessorOrder;
    /** Status to filter by, with possible values `started`, `ended` or `failed`. */
    status?: MediaProcessorStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: MediaProcessorInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface MediaProcessorListInstanceOptions {
    /** The sort order of the list by `date_created`. Can be: `asc` (ascending) or `desc` (descending) with `desc` as the default. */
    order?: MediaProcessorOrder;
    /** Status to filter by, with possible values `started`, `ended` or `failed`. */
    status?: MediaProcessorStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface MediaProcessorListInstancePageOptions {
    /** The sort order of the list by `date_created`. Can be: `asc` (ascending) or `desc` (descending) with `desc` as the default. */
    order?: MediaProcessorOrder;
    /** Status to filter by, with possible values `started`, `ended` or `failed`. */
    status?: MediaProcessorStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface MediaProcessorContext {
    /**
     * Fetch a MediaProcessorInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaProcessorInstance
     */
    fetch(callback?: (error: Error | null, item?: MediaProcessorInstance) => any): Promise<MediaProcessorInstance>;
    /**
     * Update a MediaProcessorInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaProcessorInstance
     */
    update(params: MediaProcessorContextUpdateOptions, callback?: (error: Error | null, item?: MediaProcessorInstance) => any): Promise<MediaProcessorInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface MediaProcessorContextSolution {
    sid: string;
}
export declare class MediaProcessorContextImpl implements MediaProcessorContext {
    protected _version: V1;
    protected _solution: MediaProcessorContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: MediaProcessorInstance) => any): Promise<MediaProcessorInstance>;
    update(params: MediaProcessorContextUpdateOptions, callback?: (error: Error | null, item?: MediaProcessorInstance) => any): Promise<MediaProcessorInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): MediaProcessorContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface MediaProcessorPayload extends TwilioResponsePayload {
    media_processors: MediaProcessorResource[];
}
interface MediaProcessorResource {
    account_sid: string;
    sid: string;
    date_created: Date;
    date_updated: Date;
    extension: string;
    extension_context: string;
    status: MediaProcessorStatus;
    url: string;
    ended_reason: string;
    status_callback: string;
    status_callback_method: string;
    max_duration: number;
}
export declare class MediaProcessorInstance {
    protected _version: V1;
    protected _solution: MediaProcessorContextSolution;
    protected _context?: MediaProcessorContext;
    constructor(_version: V1, payload: MediaProcessorResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the MediaProcessor resource.
     */
    accountSid: string;
    /**
     * The unique string generated to identify the MediaProcessor resource.
     */
    sid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The [Media Extension](/docs/live/media-extensions-overview) name or URL. Ex: `video-composer-v2`
     */
    extension: string;
    /**
     * The context of the Media Extension, represented as a JSON dictionary. See the documentation for the specific [Media Extension](/docs/live/media-extensions-overview) you are using for more information about the context to send.
     */
    extensionContext: string;
    status: MediaProcessorStatus;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    /**
     * The reason why a MediaProcessor ended. When a MediaProcessor is in progress, will be `null`. When a MediaProcessor is completed, can be `ended-via-api`, `max-duration-exceeded`, `error-loading-extension`, `error-streaming-media` or `internal-service-error`. See [ended reasons](/docs/live/api/mediaprocessors#mediaprocessor-ended-reason-values) for more details.
     */
    endedReason: string;
    /**
     * The URL to which Twilio will send asynchronous webhook requests for every MediaProcessor event. See [Status Callbacks](/docs/live/api/status-callbacks) for details.
     */
    statusCallback: string;
    /**
     * The HTTP method Twilio should use to call the `status_callback` URL. Can be `POST` or `GET` and the default is `POST`.
     */
    statusCallbackMethod: string;
    /**
     * The maximum time, in seconds, that the MediaProcessor can run before automatically ends. The default value is 300 seconds, and the maximum value is 90000 seconds. Once this maximum duration is reached, Twilio will end the MediaProcessor, regardless of whether media is still streaming.
     */
    maxDuration: number;
    private get _proxy();
    /**
     * Fetch a MediaProcessorInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaProcessorInstance
     */
    fetch(callback?: (error: Error | null, item?: MediaProcessorInstance) => any): Promise<MediaProcessorInstance>;
    /**
     * Update a MediaProcessorInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaProcessorInstance
     */
    update(params: MediaProcessorContextUpdateOptions, callback?: (error: Error | null, item?: MediaProcessorInstance) => any): Promise<MediaProcessorInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        sid: string;
        dateCreated: Date;
        dateUpdated: Date;
        extension: string;
        extensionContext: string;
        status: MediaProcessorStatus;
        url: string;
        endedReason: string;
        statusCallback: string;
        statusCallbackMethod: string;
        maxDuration: number;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface MediaProcessorSolution {
}
export interface MediaProcessorListInstance {
    _version: V1;
    _solution: MediaProcessorSolution;
    _uri: string;
    (sid: string): MediaProcessorContext;
    get(sid: string): MediaProcessorContext;
    /**
     * Create a MediaProcessorInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaProcessorInstance
     */
    create(params: MediaProcessorListInstanceCreateOptions, callback?: (error: Error | null, item?: MediaProcessorInstance) => any): Promise<MediaProcessorInstance>;
    /**
     * Streams MediaProcessorInstance records from the API.
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
     * @param { MediaProcessorListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: MediaProcessorInstance, done: (err?: Error) => void) => void): void;
    each(params: MediaProcessorListInstanceEachOptions, callback?: (item: MediaProcessorInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of MediaProcessorInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: MediaProcessorPage) => any): Promise<MediaProcessorPage>;
    /**
     * Lists MediaProcessorInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MediaProcessorListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: MediaProcessorInstance[]) => any): Promise<MediaProcessorInstance[]>;
    list(params: MediaProcessorListInstanceOptions, callback?: (error: Error | null, items: MediaProcessorInstance[]) => any): Promise<MediaProcessorInstance[]>;
    /**
     * Retrieve a single page of MediaProcessorInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MediaProcessorListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: MediaProcessorPage) => any): Promise<MediaProcessorPage>;
    page(params: MediaProcessorListInstancePageOptions, callback?: (error: Error | null, items: MediaProcessorPage) => any): Promise<MediaProcessorPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function MediaProcessorListInstance(version: V1): MediaProcessorListInstance;
export declare class MediaProcessorPage extends Page<V1, MediaProcessorPayload, MediaProcessorResource, MediaProcessorInstance> {
    /**
     * Initialize the MediaProcessorPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: MediaProcessorSolution);
    /**
     * Build an instance of MediaProcessorInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: MediaProcessorResource): MediaProcessorInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
