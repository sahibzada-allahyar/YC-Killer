/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type ParticipantCodec = "VP8" | "H264" | "VP9";
export type ParticipantEdgeLocation = "ashburn" | "dublin" | "frankfurt" | "singapore" | "sydney" | "sao_paulo" | "roaming" | "umatilla" | "tokyo";
export type ParticipantRoomStatus = "in_progress" | "completed";
export type ParticipantTwilioRealm = "us1" | "us2" | "au1" | "br1" | "ie1" | "jp1" | "sg1" | "in1" | "de1" | "gll";
/**
 * Options to pass to each
 */
export interface ParticipantListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ParticipantInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ParticipantListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ParticipantListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ParticipantContext {
    /**
     * Fetch a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    fetch(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ParticipantContextSolution {
    roomSid: string;
    participantSid: string;
}
export declare class ParticipantContextImpl implements ParticipantContext {
    protected _version: V1;
    protected _solution: ParticipantContextSolution;
    protected _uri: string;
    constructor(_version: V1, roomSid: string, participantSid: string);
    fetch(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ParticipantContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ParticipantPayload extends TwilioResponsePayload {
    participants: ParticipantResource[];
}
interface ParticipantResource {
    participant_sid: string;
    participant_identity: string;
    join_time: Date;
    leave_time: Date;
    duration_sec: number;
    account_sid: string;
    room_sid: string;
    status: ParticipantRoomStatus;
    codecs: Array<ParticipantCodec>;
    end_reason: string;
    error_code: number;
    error_code_url: string;
    media_region: ParticipantTwilioRealm;
    properties: any;
    edge_location: ParticipantEdgeLocation;
    publisher_info: any;
    url: string;
}
export declare class ParticipantInstance {
    protected _version: V1;
    protected _solution: ParticipantContextSolution;
    protected _context?: ParticipantContext;
    constructor(_version: V1, payload: ParticipantResource, roomSid: string, participantSid?: string);
    /**
     * Unique identifier for the participant.
     */
    participantSid: string;
    /**
     * The application-defined string that uniquely identifies the participant within a Room.
     */
    participantIdentity: string;
    /**
     * When the participant joined the room.
     */
    joinTime: Date;
    /**
     * When the participant left the room.
     */
    leaveTime: Date;
    /**
     * Amount of time in seconds the participant was in the room.
     */
    durationSec: number;
    /**
     * Account SID associated with the room.
     */
    accountSid: string;
    /**
     * Unique identifier for the room.
     */
    roomSid: string;
    status: ParticipantRoomStatus;
    /**
     * Codecs detected from the participant. Can be `VP8`, `H264`, or `VP9`.
     */
    codecs: Array<ParticipantCodec>;
    /**
     * Reason the participant left the room. See [the list of possible values here](https://www.twilio.com/docs/video/troubleshooting/video-log-analyzer-api#end_reason).
     */
    endReason: string;
    /**
     * Errors encountered by the participant.
     */
    errorCode: number;
    /**
     * Twilio error code dictionary link.
     */
    errorCodeUrl: string;
    mediaRegion: ParticipantTwilioRealm;
    /**
     * Object containing information about the participant\'s data from the room. See [below](https://www.twilio.com/docs/video/troubleshooting/video-log-analyzer-api#properties) for more information.
     */
    properties: any;
    edgeLocation: ParticipantEdgeLocation;
    /**
     * Object containing information about the SDK name and version. See [below](https://www.twilio.com/docs/video/troubleshooting/video-log-analyzer-api#publisher_info) for more information.
     */
    publisherInfo: any;
    /**
     * URL of the participant resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    fetch(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        participantSid: string;
        participantIdentity: string;
        joinTime: Date;
        leaveTime: Date;
        durationSec: number;
        accountSid: string;
        roomSid: string;
        status: ParticipantRoomStatus;
        codecs: ParticipantCodec[];
        endReason: string;
        errorCode: number;
        errorCodeUrl: string;
        mediaRegion: ParticipantTwilioRealm;
        properties: any;
        edgeLocation: ParticipantEdgeLocation;
        publisherInfo: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ParticipantSolution {
    roomSid: string;
}
export interface ParticipantListInstance {
    _version: V1;
    _solution: ParticipantSolution;
    _uri: string;
    (participantSid: string): ParticipantContext;
    get(participantSid: string): ParticipantContext;
    /**
     * Streams ParticipantInstance records from the API.
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
     * @param { ParticipantListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ParticipantInstance, done: (err?: Error) => void) => void): void;
    each(params: ParticipantListInstanceEachOptions, callback?: (item: ParticipantInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ParticipantInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ParticipantPage) => any): Promise<ParticipantPage>;
    /**
     * Lists ParticipantInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ParticipantListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ParticipantInstance[]) => any): Promise<ParticipantInstance[]>;
    list(params: ParticipantListInstanceOptions, callback?: (error: Error | null, items: ParticipantInstance[]) => any): Promise<ParticipantInstance[]>;
    /**
     * Retrieve a single page of ParticipantInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ParticipantListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ParticipantPage) => any): Promise<ParticipantPage>;
    page(params: ParticipantListInstancePageOptions, callback?: (error: Error | null, items: ParticipantPage) => any): Promise<ParticipantPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ParticipantListInstance(version: V1, roomSid: string): ParticipantListInstance;
export declare class ParticipantPage extends Page<V1, ParticipantPayload, ParticipantResource, ParticipantInstance> {
    /**
     * Initialize the ParticipantPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ParticipantSolution);
    /**
     * Build an instance of ParticipantInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ParticipantResource): ParticipantInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
