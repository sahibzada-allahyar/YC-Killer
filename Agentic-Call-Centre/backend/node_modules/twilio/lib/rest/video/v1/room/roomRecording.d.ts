/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type RoomRecordingCodec = "VP8" | "H264" | "OPUS" | "PCMU";
export type RoomRecordingFormat = "mka" | "mkv";
export type RoomRecordingStatus = "processing" | "completed" | "deleted" | "failed";
export type RoomRecordingType = "audio" | "video" | "data";
/**
 * Options to pass to each
 */
export interface RoomRecordingListInstanceEachOptions {
    /** Read only the recordings with this status. Can be: `processing`, `completed`, or `deleted`. */
    status?: RoomRecordingStatus;
    /** Read only the recordings that have this `source_sid`. */
    sourceSid?: string;
    /** Read only recordings that started on or after this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime with time zone. */
    dateCreatedAfter?: Date;
    /** Read only Recordings that started before this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime with time zone. */
    dateCreatedBefore?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: RoomRecordingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface RoomRecordingListInstanceOptions {
    /** Read only the recordings with this status. Can be: `processing`, `completed`, or `deleted`. */
    status?: RoomRecordingStatus;
    /** Read only the recordings that have this `source_sid`. */
    sourceSid?: string;
    /** Read only recordings that started on or after this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime with time zone. */
    dateCreatedAfter?: Date;
    /** Read only Recordings that started before this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime with time zone. */
    dateCreatedBefore?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface RoomRecordingListInstancePageOptions {
    /** Read only the recordings with this status. Can be: `processing`, `completed`, or `deleted`. */
    status?: RoomRecordingStatus;
    /** Read only the recordings that have this `source_sid`. */
    sourceSid?: string;
    /** Read only recordings that started on or after this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime with time zone. */
    dateCreatedAfter?: Date;
    /** Read only Recordings that started before this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime with time zone. */
    dateCreatedBefore?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface RoomRecordingContext {
    /**
     * Remove a RoomRecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RoomRecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoomRecordingInstance
     */
    fetch(callback?: (error: Error | null, item?: RoomRecordingInstance) => any): Promise<RoomRecordingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface RoomRecordingContextSolution {
    roomSid: string;
    sid: string;
}
export declare class RoomRecordingContextImpl implements RoomRecordingContext {
    protected _version: V1;
    protected _solution: RoomRecordingContextSolution;
    protected _uri: string;
    constructor(_version: V1, roomSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: RoomRecordingInstance) => any): Promise<RoomRecordingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): RoomRecordingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface RoomRecordingPayload extends TwilioResponsePayload {
    recordings: RoomRecordingResource[];
}
interface RoomRecordingResource {
    account_sid: string;
    status: RoomRecordingStatus;
    date_created: Date;
    sid: string;
    source_sid: string;
    size: number;
    url: string;
    type: RoomRecordingType;
    duration: number;
    container_format: RoomRecordingFormat;
    codec: RoomRecordingCodec;
    grouping_sids: any;
    track_name: string;
    offset: number;
    media_external_location: string;
    room_sid: string;
    links: Record<string, string>;
}
export declare class RoomRecordingInstance {
    protected _version: V1;
    protected _solution: RoomRecordingContextSolution;
    protected _context?: RoomRecordingContext;
    constructor(_version: V1, payload: RoomRecordingResource, roomSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the RoomRecording resource.
     */
    accountSid: string;
    status: RoomRecordingStatus;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The unique string that we created to identify the RoomRecording resource.
     */
    sid: string;
    /**
     * The SID of the recording source. For a Room Recording, this value is a `track_sid`.
     */
    sourceSid: string;
    /**
     * The size of the recorded track in bytes.
     */
    size: number;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    type: RoomRecordingType;
    /**
     * The duration of the recording rounded to the nearest second. Sub-second duration tracks have a `duration` of 1 second
     */
    duration: number;
    containerFormat: RoomRecordingFormat;
    codec: RoomRecordingCodec;
    /**
     * A list of SIDs related to the Recording. Includes the `room_sid` and `participant_sid`.
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
     * The SID of the Room resource the recording is associated with.
     */
    roomSid: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a RoomRecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RoomRecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoomRecordingInstance
     */
    fetch(callback?: (error: Error | null, item?: RoomRecordingInstance) => any): Promise<RoomRecordingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        status: RoomRecordingStatus;
        dateCreated: Date;
        sid: string;
        sourceSid: string;
        size: number;
        url: string;
        type: RoomRecordingType;
        duration: number;
        containerFormat: RoomRecordingFormat;
        codec: RoomRecordingCodec;
        groupingSids: any;
        trackName: string;
        offset: number;
        mediaExternalLocation: string;
        roomSid: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface RoomRecordingSolution {
    roomSid: string;
}
export interface RoomRecordingListInstance {
    _version: V1;
    _solution: RoomRecordingSolution;
    _uri: string;
    (sid: string): RoomRecordingContext;
    get(sid: string): RoomRecordingContext;
    /**
     * Streams RoomRecordingInstance records from the API.
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
     * @param { RoomRecordingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: RoomRecordingInstance, done: (err?: Error) => void) => void): void;
    each(params: RoomRecordingListInstanceEachOptions, callback?: (item: RoomRecordingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of RoomRecordingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: RoomRecordingPage) => any): Promise<RoomRecordingPage>;
    /**
     * Lists RoomRecordingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RoomRecordingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: RoomRecordingInstance[]) => any): Promise<RoomRecordingInstance[]>;
    list(params: RoomRecordingListInstanceOptions, callback?: (error: Error | null, items: RoomRecordingInstance[]) => any): Promise<RoomRecordingInstance[]>;
    /**
     * Retrieve a single page of RoomRecordingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RoomRecordingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: RoomRecordingPage) => any): Promise<RoomRecordingPage>;
    page(params: RoomRecordingListInstancePageOptions, callback?: (error: Error | null, items: RoomRecordingPage) => any): Promise<RoomRecordingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RoomRecordingListInstance(version: V1, roomSid: string): RoomRecordingListInstance;
export declare class RoomRecordingPage extends Page<V1, RoomRecordingPayload, RoomRecordingResource, RoomRecordingInstance> {
    /**
     * Initialize the RoomRecordingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: RoomRecordingSolution);
    /**
     * Build an instance of RoomRecordingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: RoomRecordingResource): RoomRecordingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
