/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { ParticipantListInstance } from "./room/participant";
export type RoomCodec = "VP8" | "H264" | "VP9";
export type RoomCreatedMethod = "sdk" | "ad_hoc" | "api";
export type RoomEdgeLocation = "ashburn" | "dublin" | "frankfurt" | "singapore" | "sydney" | "sao_paulo" | "roaming" | "umatilla" | "tokyo";
export type RoomEndReason = "room_ended_via_api" | "timeout";
export type RoomProcessingState = "complete" | "in_progress";
export type RoomRoomStatus = "in_progress" | "completed";
export type RoomRoomType = "go" | "peer_to_peer" | "group" | "group_small";
export type RoomTwilioRealm = "us1" | "us2" | "au1" | "br1" | "ie1" | "jp1" | "sg1" | "in1" | "de1" | "gll";
/**
 * Options to pass to each
 */
export interface RoomListInstanceEachOptions {
    /** Type of room. Can be `go`, `peer_to_peer`, `group`, or `group_small`. */
    roomType?: Array<RoomRoomType>;
    /** Codecs used by participants in the room. Can be `VP8`, `H264`, or `VP9`. */
    codec?: Array<RoomCodec>;
    /** Room friendly name. */
    roomName?: string;
    /** Only read rooms that started on or after this ISO 8601 timestamp. */
    createdAfter?: Date;
    /** Only read rooms that started before this ISO 8601 timestamp. */
    createdBefore?: Date;
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
    /** Type of room. Can be `go`, `peer_to_peer`, `group`, or `group_small`. */
    roomType?: Array<RoomRoomType>;
    /** Codecs used by participants in the room. Can be `VP8`, `H264`, or `VP9`. */
    codec?: Array<RoomCodec>;
    /** Room friendly name. */
    roomName?: string;
    /** Only read rooms that started on or after this ISO 8601 timestamp. */
    createdAfter?: Date;
    /** Only read rooms that started before this ISO 8601 timestamp. */
    createdBefore?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface RoomListInstancePageOptions {
    /** Type of room. Can be `go`, `peer_to_peer`, `group`, or `group_small`. */
    roomType?: Array<RoomRoomType>;
    /** Codecs used by participants in the room. Can be `VP8`, `H264`, or `VP9`. */
    codec?: Array<RoomCodec>;
    /** Room friendly name. */
    roomName?: string;
    /** Only read rooms that started on or after this ISO 8601 timestamp. */
    createdAfter?: Date;
    /** Only read rooms that started before this ISO 8601 timestamp. */
    createdBefore?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface RoomContext {
    participants: ParticipantListInstance;
    /**
     * Fetch a RoomInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoomInstance
     */
    fetch(callback?: (error: Error | null, item?: RoomInstance) => any): Promise<RoomInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface RoomContextSolution {
    roomSid: string;
}
export declare class RoomContextImpl implements RoomContext {
    protected _version: V1;
    protected _solution: RoomContextSolution;
    protected _uri: string;
    protected _participants?: ParticipantListInstance;
    constructor(_version: V1, roomSid: string);
    get participants(): ParticipantListInstance;
    fetch(callback?: (error: Error | null, item?: RoomInstance) => any): Promise<RoomInstance>;
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
    account_sid: string;
    room_sid: string;
    room_name: string;
    create_time: Date;
    end_time: Date;
    room_type: RoomRoomType;
    room_status: RoomRoomStatus;
    status_callback: string;
    status_callback_method: string;
    created_method: RoomCreatedMethod;
    end_reason: RoomEndReason;
    max_participants: number;
    unique_participants: number;
    unique_participant_identities: number;
    concurrent_participants: number;
    max_concurrent_participants: number;
    codecs: Array<RoomCodec>;
    media_region: RoomTwilioRealm;
    duration_sec: number;
    total_participant_duration_sec: number;
    total_recording_duration_sec: number;
    processing_state: RoomProcessingState;
    recording_enabled: boolean;
    edge_location: RoomEdgeLocation;
    url: string;
    links: Record<string, string>;
}
export declare class RoomInstance {
    protected _version: V1;
    protected _solution: RoomContextSolution;
    protected _context?: RoomContext;
    constructor(_version: V1, payload: RoomResource, roomSid?: string);
    /**
     * Account SID associated with this room.
     */
    accountSid: string;
    /**
     * Unique identifier for the room.
     */
    roomSid: string;
    /**
     * Room friendly name.
     */
    roomName: string;
    /**
     * Creation time of the room.
     */
    createTime: Date;
    /**
     * End time for the room.
     */
    endTime: Date;
    roomType: RoomRoomType;
    roomStatus: RoomRoomStatus;
    /**
     * Webhook provided for status callbacks.
     */
    statusCallback: string;
    /**
     * HTTP method provided for status callback URL.
     */
    statusCallbackMethod: string;
    createdMethod: RoomCreatedMethod;
    endReason: RoomEndReason;
    /**
     * Max number of total participants allowed by the application settings.
     */
    maxParticipants: number;
    /**
     * Number of participants. May include duplicate identities for participants who left and rejoined.
     */
    uniqueParticipants: number;
    /**
     * Unique number of participant identities.
     */
    uniqueParticipantIdentities: number;
    /**
     * Actual number of concurrent participants.
     */
    concurrentParticipants: number;
    /**
     * Maximum number of participants allowed in the room at the same time allowed by the application settings.
     */
    maxConcurrentParticipants: number;
    /**
     * Codecs used by participants in the room. Can be `VP8`, `H264`, or `VP9`.
     */
    codecs: Array<RoomCodec>;
    mediaRegion: RoomTwilioRealm;
    /**
     * Total room duration from create time to end time.
     */
    durationSec: number;
    /**
     * Combined amount of participant time in the room.
     */
    totalParticipantDurationSec: number;
    /**
     * Combined amount of recorded seconds for participants in the room.
     */
    totalRecordingDurationSec: number;
    processingState: RoomProcessingState;
    /**
     * Boolean indicating if recording is enabled for the room.
     */
    recordingEnabled: boolean;
    edgeLocation: RoomEdgeLocation;
    /**
     * URL for the room resource.
     */
    url: string;
    /**
     * Room subresources.
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
     * Access the participants.
     */
    participants(): ParticipantListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        roomSid: string;
        roomName: string;
        createTime: Date;
        endTime: Date;
        roomType: RoomRoomType;
        roomStatus: RoomRoomStatus;
        statusCallback: string;
        statusCallbackMethod: string;
        createdMethod: RoomCreatedMethod;
        endReason: RoomEndReason;
        maxParticipants: number;
        uniqueParticipants: number;
        uniqueParticipantIdentities: number;
        concurrentParticipants: number;
        maxConcurrentParticipants: number;
        codecs: RoomCodec[];
        mediaRegion: RoomTwilioRealm;
        durationSec: number;
        totalParticipantDurationSec: number;
        totalRecordingDurationSec: number;
        processingState: RoomProcessingState;
        recordingEnabled: boolean;
        edgeLocation: RoomEdgeLocation;
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
    (roomSid: string): RoomContext;
    get(roomSid: string): RoomContext;
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
