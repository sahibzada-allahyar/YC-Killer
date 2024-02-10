/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { ParticipantListInstance } from "./room/participant";
import { RecordingRulesListInstance } from "./room/recordingRules";
import { RoomRecordingListInstance } from "./room/roomRecording";
export type RoomRoomStatus = "in-progress" | "completed" | "failed";
export type RoomRoomType = "go" | "peer-to-peer" | "group" | "group-small";
export type RoomVideoCodec = "VP8" | "H264";
/**
 * Options to pass to update a RoomInstance
 */
export interface RoomContextUpdateOptions {
    /**  */
    status: RoomRoomStatus;
}
/**
 * Options to pass to create a RoomInstance
 */
export interface RoomListInstanceCreateOptions {
    /** Deprecated, now always considered to be true. */
    enableTurn?: boolean;
    /**  */
    type?: RoomRoomType;
    /** An application-defined string that uniquely identifies the resource. It can be used as a `room_sid` in place of the resource\\\'s `sid` in the URL to address the resource, assuming it does not contain any [reserved characters](https://tools.ietf.org/html/rfc3986#section-2.2) that would need to be URL encoded. This value is unique for `in-progress` rooms. SDK clients can use this name to connect to the room. REST API clients can use this name in place of the Room SID to interact with the room as long as the room is `in-progress`. */
    uniqueName?: string;
    /** The URL we should call using the `status_callback_method` to send status information to your application on every room event. See [Status Callbacks](https://www.twilio.com/docs/video/api/status-callbacks) for more info. */
    statusCallback?: string;
    /** The HTTP method we should use to call `status_callback`. Can be `POST` or `GET`. */
    statusCallbackMethod?: string;
    /** The maximum number of concurrent Participants allowed in the room. Peer-to-peer rooms can have up to 10 Participants. Small Group rooms can have up to 4 Participants. Group rooms can have up to 50 Participants. */
    maxParticipants?: number;
    /** Whether to start recording when Participants connect. ***This feature is not available in `peer-to-peer` rooms.*** */
    recordParticipantsOnConnect?: boolean;
    /** An array of the video codecs that are supported when publishing a track in the room.  Can be: `VP8` and `H264`.  ***This feature is not available in `peer-to-peer` rooms*** */
    videoCodecs?: Array<RoomVideoCodec>;
    /** The region for the media server in Group Rooms.  Can be: one of the [available Media Regions](https://www.twilio.com/docs/video/ip-addresses#group-rooms-media-servers). ***This feature is not available in `peer-to-peer` rooms.*** */
    mediaRegion?: string;
    /** A collection of Recording Rules that describe how to include or exclude matching tracks for recording */
    recordingRules?: any;
    /** When set to true, indicates that the participants in the room will only publish audio. No video tracks will be allowed. Group rooms only. */
    audioOnly?: boolean;
    /** The maximum number of seconds a Participant can be connected to the room. The maximum possible value is 86400 seconds (24 hours). The default is 14400 seconds (4 hours). */
    maxParticipantDuration?: number;
    /** Configures how long (in minutes) a room will remain active after last participant leaves. Valid values range from 1 to 60 minutes (no fractions). */
    emptyRoomTimeout?: number;
    /** Configures how long (in minutes) a room will remain active if no one joins. Valid values range from 1 to 60 minutes (no fractions). */
    unusedRoomTimeout?: number;
    /** When set to true, indicated that this is the large room. */
    largeRoom?: boolean;
}
/**
 * Options to pass to each
 */
export interface RoomListInstanceEachOptions {
    /** Read only the rooms with this status. Can be: `in-progress` (default) or `completed` */
    status?: RoomRoomStatus;
    /** Read only rooms with the this `unique_name`. */
    uniqueName?: string;
    /** Read only rooms that started on or after this date, given as `YYYY-MM-DD`. */
    dateCreatedAfter?: Date;
    /** Read only rooms that started before this date, given as `YYYY-MM-DD`. */
    dateCreatedBefore?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: RoomInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface RoomListInstanceOptions {
    /** Read only the rooms with this status. Can be: `in-progress` (default) or `completed` */
    status?: RoomRoomStatus;
    /** Read only rooms with the this `unique_name`. */
    uniqueName?: string;
    /** Read only rooms that started on or after this date, given as `YYYY-MM-DD`. */
    dateCreatedAfter?: Date;
    /** Read only rooms that started before this date, given as `YYYY-MM-DD`. */
    dateCreatedBefore?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface RoomListInstancePageOptions {
    /** Read only the rooms with this status. Can be: `in-progress` (default) or `completed` */
    status?: RoomRoomStatus;
    /** Read only rooms with the this `unique_name`. */
    uniqueName?: string;
    /** Read only rooms that started on or after this date, given as `YYYY-MM-DD`. */
    dateCreatedAfter?: Date;
    /** Read only rooms that started before this date, given as `YYYY-MM-DD`. */
    dateCreatedBefore?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface RoomContext {
    participants: ParticipantListInstance;
    recordingRules: RecordingRulesListInstance;
    recordings: RoomRecordingListInstance;
    /**
     * Fetch a RoomInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoomInstance
     */
    fetch(callback?: (error: Error | null, item?: RoomInstance) => any): Promise<RoomInstance>;
    /**
     * Update a RoomInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoomInstance
     */
    update(params: RoomContextUpdateOptions, callback?: (error: Error | null, item?: RoomInstance) => any): Promise<RoomInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface RoomContextSolution {
    sid: string;
}
export declare class RoomContextImpl implements RoomContext {
    protected _version: V1;
    protected _solution: RoomContextSolution;
    protected _uri: string;
    protected _participants?: ParticipantListInstance;
    protected _recordingRules?: RecordingRulesListInstance;
    protected _recordings?: RoomRecordingListInstance;
    constructor(_version: V1, sid: string);
    get participants(): ParticipantListInstance;
    get recordingRules(): RecordingRulesListInstance;
    get recordings(): RoomRecordingListInstance;
    fetch(callback?: (error: Error | null, item?: RoomInstance) => any): Promise<RoomInstance>;
    update(params: RoomContextUpdateOptions, callback?: (error: Error | null, item?: RoomInstance) => any): Promise<RoomInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): RoomContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface RoomPayload extends TwilioResponsePayload {
    rooms: RoomResource[];
}
interface RoomResource {
    sid: string;
    status: RoomRoomStatus;
    date_created: Date;
    date_updated: Date;
    account_sid: string;
    enable_turn: boolean;
    unique_name: string;
    status_callback: string;
    status_callback_method: string;
    end_time: Date;
    duration: number;
    type: RoomRoomType;
    max_participants: number;
    max_participant_duration: number;
    max_concurrent_published_tracks: number;
    record_participants_on_connect: boolean;
    video_codecs: Array<RoomVideoCodec>;
    media_region: string;
    audio_only: boolean;
    empty_room_timeout: number;
    unused_room_timeout: number;
    large_room: boolean;
    url: string;
    links: Record<string, string>;
}
export declare class RoomInstance {
    protected _version: V1;
    protected _solution: RoomContextSolution;
    protected _context?: RoomContext;
    constructor(_version: V1, payload: RoomResource, sid?: string);
    /**
     * The unique string that we created to identify the Room resource.
     */
    sid: string;
    status: RoomRoomStatus;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Room resource.
     */
    accountSid: string;
    /**
     * Deprecated, now always considered to be true.
     */
    enableTurn: boolean;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used as a `room_sid` in place of the resource\'s `sid` in the URL to address the resource, assuming it does not contain any [reserved characters](https://tools.ietf.org/html/rfc3986#section-2.2) that would need to be URL encoded. This value is unique for `in-progress` rooms. SDK clients can use this name to connect to the room. REST API clients can use this name in place of the Room SID to interact with the room as long as the room is `in-progress`.
     */
    uniqueName: string;
    /**
     * The URL we call using the `status_callback_method` to send status information to your application on every room event. See [Status Callbacks](https://www.twilio.com/docs/video/api/status-callbacks) for more info.
     */
    statusCallback: string;
    /**
     * The HTTP method we use to call `status_callback`. Can be `POST` or `GET` and defaults to `POST`.
     */
    statusCallbackMethod: string;
    /**
     * The UTC end time of the room in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#UTC) format.
     */
    endTime: Date;
    /**
     * The duration of the room in seconds.
     */
    duration: number;
    type: RoomRoomType;
    /**
     * The maximum number of concurrent Participants allowed in the room.
     */
    maxParticipants: number;
    /**
     * The maximum number of seconds a Participant can be connected to the room. The maximum possible value is 86400 seconds (24 hours). The default is 14400 seconds (4 hours).
     */
    maxParticipantDuration: number;
    /**
     * The maximum number of published audio, video, and data tracks all participants combined are allowed to publish in the room at the same time. Check [Programmable Video Limits](https://www.twilio.com/docs/video/programmable-video-limits) for more details. If it is set to 0 it means unconstrained.
     */
    maxConcurrentPublishedTracks: number;
    /**
     * Whether to start recording when Participants connect. ***This feature is not available in `peer-to-peer` rooms.***
     */
    recordParticipantsOnConnect: boolean;
    /**
     * An array of the video codecs that are supported when publishing a track in the room.  Can be: `VP8` and `H264`.  ***This feature is not available in `peer-to-peer` rooms***
     */
    videoCodecs: Array<RoomVideoCodec>;
    /**
     * The region for the media server in Group Rooms.  Can be: one of the [available Media Regions](https://www.twilio.com/docs/video/ip-addresses#media-servers). ***This feature is not available in `peer-to-peer` rooms.***
     */
    mediaRegion: string;
    /**
     * When set to true, indicates that the participants in the room will only publish audio. No video tracks will be allowed. Group rooms only.
     */
    audioOnly: boolean;
    /**
     * Specifies how long (in minutes) a room will remain active after last participant leaves. Can be configured when creating a room via REST API. For Ad-Hoc rooms this value cannot be changed.
     */
    emptyRoomTimeout: number;
    /**
     * Specifies how long (in minutes) a room will remain active if no one joins. Can be configured when creating a room via REST API. For Ad-Hoc rooms this value cannot be changed.
     */
    unusedRoomTimeout: number;
    /**
     * Indicates if this is a large room.
     */
    largeRoom: boolean;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a RoomInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoomInstance
     */
    fetch(callback?: (error: Error | null, item?: RoomInstance) => any): Promise<RoomInstance>;
    /**
     * Update a RoomInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoomInstance
     */
    update(params: RoomContextUpdateOptions, callback?: (error: Error | null, item?: RoomInstance) => any): Promise<RoomInstance>;
    /**
     * Access the participants.
     */
    participants(): ParticipantListInstance;
    /**
     * Access the recordingRules.
     */
    recordingRules(): RecordingRulesListInstance;
    /**
     * Access the recordings.
     */
    recordings(): RoomRecordingListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        status: RoomRoomStatus;
        dateCreated: Date;
        dateUpdated: Date;
        accountSid: string;
        enableTurn: boolean;
        uniqueName: string;
        statusCallback: string;
        statusCallbackMethod: string;
        endTime: Date;
        duration: number;
        type: RoomRoomType;
        maxParticipants: number;
        maxParticipantDuration: number;
        maxConcurrentPublishedTracks: number;
        recordParticipantsOnConnect: boolean;
        videoCodecs: RoomVideoCodec[];
        mediaRegion: string;
        audioOnly: boolean;
        emptyRoomTimeout: number;
        unusedRoomTimeout: number;
        largeRoom: boolean;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface RoomSolution {
}
export interface RoomListInstance {
    _version: V1;
    _solution: RoomSolution;
    _uri: string;
    (sid: string): RoomContext;
    get(sid: string): RoomContext;
    /**
     * Create a RoomInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoomInstance
     */
    create(callback?: (error: Error | null, item?: RoomInstance) => any): Promise<RoomInstance>;
    /**
     * Create a RoomInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoomInstance
     */
    create(params: RoomListInstanceCreateOptions, callback?: (error: Error | null, item?: RoomInstance) => any): Promise<RoomInstance>;
    /**
     * Streams RoomInstance records from the API.
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
     * @param { RoomListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: RoomInstance, done: (err?: Error) => void) => void): void;
    each(params: RoomListInstanceEachOptions, callback?: (item: RoomInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of RoomInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: RoomPage) => any): Promise<RoomPage>;
    /**
     * Lists RoomInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RoomListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: RoomInstance[]) => any): Promise<RoomInstance[]>;
    list(params: RoomListInstanceOptions, callback?: (error: Error | null, items: RoomInstance[]) => any): Promise<RoomInstance[]>;
    /**
     * Retrieve a single page of RoomInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RoomListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: RoomPage) => any): Promise<RoomPage>;
    page(params: RoomListInstancePageOptions, callback?: (error: Error | null, items: RoomPage) => any): Promise<RoomPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RoomListInstance(version: V1): RoomListInstance;
export declare class RoomPage extends Page<V1, RoomPayload, RoomResource, RoomInstance> {
    /**
     * Initialize the RoomPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: RoomSolution);
    /**
     * Build an instance of RoomInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: RoomResource): RoomInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
