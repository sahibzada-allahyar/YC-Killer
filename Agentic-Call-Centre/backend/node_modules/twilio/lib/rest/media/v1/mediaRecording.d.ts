/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type MediaRecordingFormat = "mp4" | "webm";
export type MediaRecordingOrder = "asc" | "desc";
export type MediaRecordingStatus = "processing" | "completed" | "deleted" | "failed";
/**
 * Options to pass to each
 */
export interface MediaRecordingListInstanceEachOptions {
    /** The sort order of the list by `date_created`. Can be: `asc` (ascending) or `desc` (descending) with `desc` as the default. */
    order?: MediaRecordingOrder;
    /** Status to filter by, with possible values `processing`, `completed`, `deleted`, or `failed`. */
    status?: MediaRecordingStatus;
    /** SID of a MediaProcessor to filter by. */
    processorSid?: string;
    /** SID of a MediaRecording source to filter by. */
    sourceSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: MediaRecordingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface MediaRecordingListInstanceOptions {
    /** The sort order of the list by `date_created`. Can be: `asc` (ascending) or `desc` (descending) with `desc` as the default. */
    order?: MediaRecordingOrder;
    /** Status to filter by, with possible values `processing`, `completed`, `deleted`, or `failed`. */
    status?: MediaRecordingStatus;
    /** SID of a MediaProcessor to filter by. */
    processorSid?: string;
    /** SID of a MediaRecording source to filter by. */
    sourceSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface MediaRecordingListInstancePageOptions {
    /** The sort order of the list by `date_created`. Can be: `asc` (ascending) or `desc` (descending) with `desc` as the default. */
    order?: MediaRecordingOrder;
    /** Status to filter by, with possible values `processing`, `completed`, `deleted`, or `failed`. */
    status?: MediaRecordingStatus;
    /** SID of a MediaProcessor to filter by. */
    processorSid?: string;
    /** SID of a MediaRecording source to filter by. */
    sourceSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface MediaRecordingContext {
    /**
     * Remove a MediaRecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a MediaRecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaRecordingInstance
     */
    fetch(callback?: (error: Error | null, item?: MediaRecordingInstance) => any): Promise<MediaRecordingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface MediaRecordingContextSolution {
    sid: string;
}
export declare class MediaRecordingContextImpl implements MediaRecordingContext {
    protected _version: V1;
    protected _solution: MediaRecordingContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: MediaRecordingInstance) => any): Promise<MediaRecordingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): MediaRecordingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface MediaRecordingPayload extends TwilioResponsePayload {
    media_recordings: MediaRecordingResource[];
}
interface MediaRecordingResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    duration: number;
    format: MediaRecordingFormat;
    links: Record<string, string>;
    processor_sid: string;
    resolution: string;
    source_sid: string;
    sid: string;
    media_size: number;
    status: MediaRecordingStatus;
    status_callback: string;
    status_callback_method: string;
    url: string;
}
export declare class MediaRecordingInstance {
    protected _version: V1;
    protected _solution: MediaRecordingContextSolution;
    protected _context?: MediaRecordingContext;
    constructor(_version: V1, payload: MediaRecordingResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the MediaRecording resource.
     */
    accountSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The duration of the MediaRecording in seconds.
     */
    duration: number;
    format: MediaRecordingFormat;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    /**
     * The SID of the MediaProcessor resource which produced the MediaRecording.
     */
    processorSid: string;
    /**
     * The dimensions of the video image in pixels expressed as columns (width) and rows (height).
     */
    resolution: string;
    /**
     * The SID of the resource that generated the original media track(s) of the MediaRecording.
     */
    sourceSid: string;
    /**
     * The unique string generated to identify the MediaRecording resource.
     */
    sid: string;
    /**
     * The size of the recording media in bytes.
     */
    mediaSize: number;
    status: MediaRecordingStatus;
    /**
     * The URL to which Twilio will send asynchronous webhook requests for every MediaRecording event. See [Status Callbacks](/docs/live/api/status-callbacks) for more details.
     */
    statusCallback: string;
    /**
     * The HTTP method Twilio should use to call the `status_callback` URL. Can be `POST` or `GET` and the default is `POST`.
     */
    statusCallbackMethod: string;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a MediaRecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a MediaRecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaRecordingInstance
     */
    fetch(callback?: (error: Error | null, item?: MediaRecordingInstance) => any): Promise<MediaRecordingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        duration: number;
        format: MediaRecordingFormat;
        links: Record<string, string>;
        processorSid: string;
        resolution: string;
        sourceSid: string;
        sid: string;
        mediaSize: number;
        status: MediaRecordingStatus;
        statusCallback: string;
        statusCallbackMethod: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface MediaRecordingSolution {
}
export interface MediaRecordingListInstance {
    _version: V1;
    _solution: MediaRecordingSolution;
    _uri: string;
    (sid: string): MediaRecordingContext;
    get(sid: string): MediaRecordingContext;
    /**
     * Streams MediaRecordingInstance records from the API.
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
     * @param { MediaRecordingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: MediaRecordingInstance, done: (err?: Error) => void) => void): void;
    each(params: MediaRecordingListInstanceEachOptions, callback?: (item: MediaRecordingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of MediaRecordingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: MediaRecordingPage) => any): Promise<MediaRecordingPage>;
    /**
     * Lists MediaRecordingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MediaRecordingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: MediaRecordingInstance[]) => any): Promise<MediaRecordingInstance[]>;
    list(params: MediaRecordingListInstanceOptions, callback?: (error: Error | null, items: MediaRecordingInstance[]) => any): Promise<MediaRecordingInstance[]>;
    /**
     * Retrieve a single page of MediaRecordingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MediaRecordingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: MediaRecordingPage) => any): Promise<MediaRecordingPage>;
    page(params: MediaRecordingListInstancePageOptions, callback?: (error: Error | null, items: MediaRecordingPage) => any): Promise<MediaRecordingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function MediaRecordingListInstance(version: V1): MediaRecordingListInstance;
export declare class MediaRecordingPage extends Page<V1, MediaRecordingPayload, MediaRecordingResource, MediaRecordingInstance> {
    /**
     * Initialize the MediaRecordingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: MediaRecordingSolution);
    /**
     * Build an instance of MediaRecordingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: MediaRecordingResource): MediaRecordingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
