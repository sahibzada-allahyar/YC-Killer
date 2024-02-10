/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { PlaybackGrantListInstance } from "./playerStreamer/playbackGrant";
export type PlayerStreamerEndedReason = "ended-via-api" | "max-duration-exceeded" | "stream-disconnected-by-source" | "unexpected-failure";
export type PlayerStreamerOrder = "asc" | "desc";
export type PlayerStreamerStatus = "created" | "started" | "ended" | "failed";
export type PlayerStreamerUpdateStatus = "ended";
/**
 * Options to pass to update a PlayerStreamerInstance
 */
export interface PlayerStreamerContextUpdateOptions {
    /**  */
    status: PlayerStreamerUpdateStatus;
}
/**
 * Options to pass to create a PlayerStreamerInstance
 */
export interface PlayerStreamerListInstanceCreateOptions {
    /** Specifies whether the PlayerStreamer is configured to stream video. Defaults to `true`. */
    video?: boolean;
    /** The URL to which Twilio will send asynchronous webhook requests for every PlayerStreamer event. See [Status Callbacks](/docs/live/api/status-callbacks) for more details. */
    statusCallback?: string;
    /** The HTTP method Twilio should use to call the `status_callback` URL. Can be `POST` or `GET` and the default is `POST`. */
    statusCallbackMethod?: string;
    /** The maximum time, in seconds, that the PlayerStreamer is active (`created` or `started`) before automatically ends. The default value is 300 seconds, and the maximum value is 90000 seconds. Once this maximum duration is reached, Twilio will end the PlayerStreamer, regardless of whether media is still streaming. */
    maxDuration?: number;
}
/**
 * Options to pass to each
 */
export interface PlayerStreamerListInstanceEachOptions {
    /** The sort order of the list by `date_created`. Can be: `asc` (ascending) or `desc` (descending) with `desc` as the default. */
    order?: PlayerStreamerOrder;
    /** Status to filter by, with possible values `created`, `started`, `ended`, or `failed`. */
    status?: PlayerStreamerStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: PlayerStreamerInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface PlayerStreamerListInstanceOptions {
    /** The sort order of the list by `date_created`. Can be: `asc` (ascending) or `desc` (descending) with `desc` as the default. */
    order?: PlayerStreamerOrder;
    /** Status to filter by, with possible values `created`, `started`, `ended`, or `failed`. */
    status?: PlayerStreamerStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface PlayerStreamerListInstancePageOptions {
    /** The sort order of the list by `date_created`. Can be: `asc` (ascending) or `desc` (descending) with `desc` as the default. */
    order?: PlayerStreamerOrder;
    /** Status to filter by, with possible values `created`, `started`, `ended`, or `failed`. */
    status?: PlayerStreamerStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface PlayerStreamerContext {
    playbackGrant: PlaybackGrantListInstance;
    /**
     * Fetch a PlayerStreamerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PlayerStreamerInstance
     */
    fetch(callback?: (error: Error | null, item?: PlayerStreamerInstance) => any): Promise<PlayerStreamerInstance>;
    /**
     * Update a PlayerStreamerInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PlayerStreamerInstance
     */
    update(params: PlayerStreamerContextUpdateOptions, callback?: (error: Error | null, item?: PlayerStreamerInstance) => any): Promise<PlayerStreamerInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PlayerStreamerContextSolution {
    sid: string;
}
export declare class PlayerStreamerContextImpl implements PlayerStreamerContext {
    protected _version: V1;
    protected _solution: PlayerStreamerContextSolution;
    protected _uri: string;
    protected _playbackGrant?: PlaybackGrantListInstance;
    constructor(_version: V1, sid: string);
    get playbackGrant(): PlaybackGrantListInstance;
    fetch(callback?: (error: Error | null, item?: PlayerStreamerInstance) => any): Promise<PlayerStreamerInstance>;
    update(params: PlayerStreamerContextUpdateOptions, callback?: (error: Error | null, item?: PlayerStreamerInstance) => any): Promise<PlayerStreamerInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PlayerStreamerContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PlayerStreamerPayload extends TwilioResponsePayload {
    player_streamers: PlayerStreamerResource[];
}
interface PlayerStreamerResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    video: boolean;
    links: Record<string, string>;
    sid: string;
    status: PlayerStreamerStatus;
    url: string;
    status_callback: string;
    status_callback_method: string;
    ended_reason: PlayerStreamerEndedReason;
    max_duration: number;
}
export declare class PlayerStreamerInstance {
    protected _version: V1;
    protected _solution: PlayerStreamerContextSolution;
    protected _context?: PlayerStreamerContext;
    constructor(_version: V1, payload: PlayerStreamerResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the PlayerStreamer resource.
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
     * Specifies whether the PlayerStreamer is configured to stream video. Defaults to `true`.
     */
    video: boolean;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    /**
     * The unique string generated to identify the PlayerStreamer resource.
     */
    sid: string;
    status: PlayerStreamerStatus;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    /**
     * The URL to which Twilio will send asynchronous webhook requests for every PlayerStreamer event. See [Status Callbacks](/docs/live/api/status-callbacks) for more details.
     */
    statusCallback: string;
    /**
     * The HTTP method Twilio should use to call the `status_callback` URL. Can be `POST` or `GET` and the default is `POST`.
     */
    statusCallbackMethod: string;
    endedReason: PlayerStreamerEndedReason;
    /**
     * The maximum time, in seconds, that the PlayerStreamer is active (`created` or `started`) before automatically ends. The default value is 300 seconds, and the maximum value is 90000 seconds. Once this maximum duration is reached, Twilio will end the PlayerStreamer, regardless of whether media is still streaming.
     */
    maxDuration: number;
    private get _proxy();
    /**
     * Fetch a PlayerStreamerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PlayerStreamerInstance
     */
    fetch(callback?: (error: Error | null, item?: PlayerStreamerInstance) => any): Promise<PlayerStreamerInstance>;
    /**
     * Update a PlayerStreamerInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PlayerStreamerInstance
     */
    update(params: PlayerStreamerContextUpdateOptions, callback?: (error: Error | null, item?: PlayerStreamerInstance) => any): Promise<PlayerStreamerInstance>;
    /**
     * Access the playbackGrant.
     */
    playbackGrant(): PlaybackGrantListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        video: boolean;
        links: Record<string, string>;
        sid: string;
        status: PlayerStreamerStatus;
        url: string;
        statusCallback: string;
        statusCallbackMethod: string;
        endedReason: PlayerStreamerEndedReason;
        maxDuration: number;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PlayerStreamerSolution {
}
export interface PlayerStreamerListInstance {
    _version: V1;
    _solution: PlayerStreamerSolution;
    _uri: string;
    (sid: string): PlayerStreamerContext;
    get(sid: string): PlayerStreamerContext;
    /**
     * Create a PlayerStreamerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PlayerStreamerInstance
     */
    create(callback?: (error: Error | null, item?: PlayerStreamerInstance) => any): Promise<PlayerStreamerInstance>;
    /**
     * Create a PlayerStreamerInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PlayerStreamerInstance
     */
    create(params: PlayerStreamerListInstanceCreateOptions, callback?: (error: Error | null, item?: PlayerStreamerInstance) => any): Promise<PlayerStreamerInstance>;
    /**
     * Streams PlayerStreamerInstance records from the API.
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
     * @param { PlayerStreamerListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: PlayerStreamerInstance, done: (err?: Error) => void) => void): void;
    each(params: PlayerStreamerListInstanceEachOptions, callback?: (item: PlayerStreamerInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of PlayerStreamerInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: PlayerStreamerPage) => any): Promise<PlayerStreamerPage>;
    /**
     * Lists PlayerStreamerInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PlayerStreamerListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: PlayerStreamerInstance[]) => any): Promise<PlayerStreamerInstance[]>;
    list(params: PlayerStreamerListInstanceOptions, callback?: (error: Error | null, items: PlayerStreamerInstance[]) => any): Promise<PlayerStreamerInstance[]>;
    /**
     * Retrieve a single page of PlayerStreamerInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PlayerStreamerListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: PlayerStreamerPage) => any): Promise<PlayerStreamerPage>;
    page(params: PlayerStreamerListInstancePageOptions, callback?: (error: Error | null, items: PlayerStreamerPage) => any): Promise<PlayerStreamerPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PlayerStreamerListInstance(version: V1): PlayerStreamerListInstance;
export declare class PlayerStreamerPage extends Page<V1, PlayerStreamerPayload, PlayerStreamerResource, PlayerStreamerInstance> {
    /**
     * Initialize the PlayerStreamerPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: PlayerStreamerSolution);
    /**
     * Build an instance of PlayerStreamerInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: PlayerStreamerResource): PlayerStreamerInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
