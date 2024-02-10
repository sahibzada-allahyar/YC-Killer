/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type RecordingCodec = "VP8" | "H264" | "OPUS" | "PCMU";
export type RecordingFormat = "mka" | "mkv";
export type RecordingStatus = "processing" | "completed" | "deleted" | "failed";
export type RecordingType = "audio" | "video" | "data";
/**
 * Options to pass to each
 */
export interface RecordingListInstanceEachOptions {
    /** Read only the recordings that have this status. Can be: `processing`, `completed`, or `deleted`. */
    status?: RecordingStatus;
    /** Read only the recordings that have this `source_sid`. */
    sourceSid?: string;
    /** Read only recordings with this `grouping_sid`, which may include a `participant_sid` and/or a `room_sid`. */
    groupingSid?: Array<string>;
    /** Read only recordings that started on or after this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time with time zone. */
    dateCreatedAfter?: Date;
    /** Read only recordings that started before this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time with time zone, given as `YYYY-MM-DDThh:mm:ss+|-hh:mm` or `YYYY-MM-DDThh:mm:ssZ`. */
    dateCreatedBefore?: Date;
    /** Read only recordings that have this media type. Can be either `audio` or `video`. */
    mediaType?: RecordingType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: RecordingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface RecordingListInstanceOptions {
    /** Read only the recordings that have this status. Can be: `processing`, `completed`, or `deleted`. */
    status?: RecordingStatus;
    /** Read only the recordings that have this `source_sid`. */
    sourceSid?: string;
    /** Read only recordings with this `grouping_sid`, which may include a `participant_sid` and/or a `room_sid`. */
    groupingSid?: Array<string>;
    /** Read only recordings that started on or after this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time with time zone. */
    dateCreatedAfter?: Date;
    /** Read only recordings that started before this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time with time zone, given as `YYYY-MM-DDThh:mm:ss+|-hh:mm` or `YYYY-MM-DDThh:mm:ssZ`. */
    dateCreatedBefore?: Date;
    /** Read only recordings that have this media type. Can be either `audio` or `video`. */
    mediaType?: RecordingType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface RecordingListInstancePageOptions {
    /** Read only the recordings that have this status. Can be: `processing`, `completed`, or `deleted`. */
    status?: RecordingStatus;
    /** Read only the recordings that have this `source_sid`. */
    sourceSid?: string;
    /** Read only recordings with this `grouping_sid`, which may include a `participant_sid` and/or a `room_sid`. */
    groupingSid?: Array<string>;
    /** Read only recordings that started on or after this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time with time zone. */
    dateCreatedAfter?: Date;
    /** Read only recordings that started before this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time with time zone, given as `YYYY-MM-DDThh:mm:ss+|-hh:mm` or `YYYY-MM-DDThh:mm:ssZ`. */
    dateCreatedBefore?: Date;
    /** Read only recordings that have this media type. Can be either `audio` or `video`. */
    mediaType?: RecordingType;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface RecordingContext {
    /**
     * Remove a RecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingInstance
     */
    fetch(callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface RecordingContextSolution {
    sid: string;
}
export declare class RecordingContextImpl implements RecordingContext {
    protected _version: V1;
    protected _solution: RecordingContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): RecordingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface RecordingPayload extends TwilioResponsePayload {
    recordings: RecordingResource[];
}
interface RecordingResource {
    account_sid: string;
    status: RecordingStatus;
    date_created: Date;
    sid: string;
    source_sid: string;
    size: number;
    url: string;
    type: RecordingType;
    duration: number;
    container_format: RecordingFormat;
    codec: RecordingCodec;
    grouping_sids: any;
    track_name: string;
    offset: number;
    media_external_location: string;
    status_callback: string;
    status_callback_method: string;
    links: Record<string, string>;
}
export declare class RecordingInstance {
    protected _version: V1;
    protected _solution: RecordingContextSolution;
    protected _context?: RecordingContext;
    constructor(_version: V1, payload: RecordingResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Recording resource.
     */
    accountSid: string;
    status: RecordingStatus;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The unique string that we created to identify the Recording resource.
     */
    sid: string;
    /**
     * The SID of the recording source. For a Room Recording, this value is a `track_sid`.
     */
    sourceSid: string;
    /**
     * The size of the recorded track, in bytes.
     */
    size: number;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    type: RecordingType;
    /**
     * The duration of the recording in seconds rounded to the nearest second. Sub-second tracks have a `Duration` property of 1 second
     */
    duration: number;
    containerFormat: RecordingFormat;
    codec: RecordingCodec;
    /**
     * A list of SIDs related to the recording. Includes the `room_sid` and `participant_sid`.
     */
    groupingSids: any;
    /**
     * The name that was given to the source track of the recording. If no name is given, the `source_sid` is used.
     */
    trackName: string;
    /**
     * The time in milliseconds elapsed between an arbitrary point in time, common to all group rooms, and the moment when the source room of this track started. This information provides a synchronization mechanism for recordings belonging to the same room.
     */
    offset: number;
    /**
     * The URL of the media file associated with the recording when stored externally. See [External S3 Recordings](/docs/video/api/external-s3-recordings) for more details.
     */
    mediaExternalLocation: string;
    /**
     * The URL called using the `status_callback_method` to send status information on every recording event.
     */
    statusCallback: string;
    /**
     * The HTTP method used to call `status_callback`. Can be: `POST` or `GET`, defaults to `POST`.
     */
    statusCallbackMethod: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a RecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingInstance
     */
    fetch(callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        status: RecordingStatus;
        dateCreated: Date;
        sid: string;
        sourceSid: string;
        size: number;
        url: string;
        type: RecordingType;
        duration: number;
        containerFormat: RecordingFormat;
        codec: RecordingCodec;
        groupingSids: any;
        trackName: string;
        offset: number;
        mediaExternalLocation: string;
        statusCallback: string;
        statusCallbackMethod: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface RecordingSolution {
}
export interface RecordingListInstance {
    _version: V1;
    _solution: RecordingSolution;
    _uri: string;
    (sid: string): RecordingContext;
    get(sid: string): RecordingContext;
    /**
     * Streams RecordingInstance records from the API.
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
     * @param { RecordingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: RecordingInstance, done: (err?: Error) => void) => void): void;
    each(params: RecordingListInstanceEachOptions, callback?: (item: RecordingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of RecordingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: RecordingPage) => any): Promise<RecordingPage>;
    /**
     * Lists RecordingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RecordingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: RecordingInstance[]) => any): Promise<RecordingInstance[]>;
    list(params: RecordingListInstanceOptions, callback?: (error: Error | null, items: RecordingInstance[]) => any): Promise<RecordingInstance[]>;
    /**
     * Retrieve a single page of RecordingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RecordingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: RecordingPage) => any): Promise<RecordingPage>;
    page(params: RecordingListInstancePageOptions, callback?: (error: Error | null, items: RecordingPage) => any): Promise<RecordingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RecordingListInstance(version: V1): RecordingListInstance;
export declare class RecordingPage extends Page<V1, RecordingPayload, RecordingResource, RecordingInstance> {
    /**
     * Initialize the RecordingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: RecordingSolution);
    /**
     * Build an instance of RecordingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: RecordingResource): RecordingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
