/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
export type SubscribedTrackKind = "audio" | "video" | "data";
/**
 * Options to pass to each
 */
export interface SubscribedTrackListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SubscribedTrackInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SubscribedTrackListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SubscribedTrackListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SubscribedTrackContext {
    /**
     * Fetch a SubscribedTrackInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscribedTrackInstance
     */
    fetch(callback?: (error: Error | null, item?: SubscribedTrackInstance) => any): Promise<SubscribedTrackInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SubscribedTrackContextSolution {
    roomSid: string;
    participantSid: string;
    sid: string;
}
export declare class SubscribedTrackContextImpl implements SubscribedTrackContext {
    protected _version: V1;
    protected _solution: SubscribedTrackContextSolution;
    protected _uri: string;
    constructor(_version: V1, roomSid: string, participantSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: SubscribedTrackInstance) => any): Promise<SubscribedTrackInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SubscribedTrackContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SubscribedTrackPayload extends TwilioResponsePayload {
    subscribed_tracks: SubscribedTrackResource[];
}
interface SubscribedTrackResource {
    sid: string;
    participant_sid: string;
    publisher_sid: string;
    room_sid: string;
    name: string;
    date_created: Date;
    date_updated: Date;
    enabled: boolean;
    kind: SubscribedTrackKind;
    url: string;
}
export declare class SubscribedTrackInstance {
    protected _version: V1;
    protected _solution: SubscribedTrackContextSolution;
    protected _context?: SubscribedTrackContext;
    constructor(_version: V1, payload: SubscribedTrackResource, roomSid: string, participantSid: string, sid?: string);
    /**
     * The unique string that we created to identify the RoomParticipantSubscribedTrack resource.
     */
    sid: string;
    /**
     * The SID of the participant that subscribes to the track.
     */
    participantSid: string;
    /**
     * The SID of the participant that publishes the track.
     */
    publisherSid: string;
    /**
     * The SID of the room where the track is published.
     */
    roomSid: string;
    /**
     * The track name. Must have no more than 128 characters and be unique among the participant\'s published tracks.
     */
    name: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * Whether the track is enabled.
     */
    enabled: boolean;
    kind: SubscribedTrackKind;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a SubscribedTrackInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscribedTrackInstance
     */
    fetch(callback?: (error: Error | null, item?: SubscribedTrackInstance) => any): Promise<SubscribedTrackInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        participantSid: string;
        publisherSid: string;
        roomSid: string;
        name: string;
        dateCreated: Date;
        dateUpdated: Date;
        enabled: boolean;
        kind: SubscribedTrackKind;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SubscribedTrackSolution {
    roomSid: string;
    participantSid: string;
}
export interface SubscribedTrackListInstance {
    _version: V1;
    _solution: SubscribedTrackSolution;
    _uri: string;
    (sid: string): SubscribedTrackContext;
    get(sid: string): SubscribedTrackContext;
    /**
     * Streams SubscribedTrackInstance records from the API.
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
     * @param { SubscribedTrackListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SubscribedTrackInstance, done: (err?: Error) => void) => void): void;
    each(params: SubscribedTrackListInstanceEachOptions, callback?: (item: SubscribedTrackInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SubscribedTrackInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SubscribedTrackPage) => any): Promise<SubscribedTrackPage>;
    /**
     * Lists SubscribedTrackInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SubscribedTrackListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SubscribedTrackInstance[]) => any): Promise<SubscribedTrackInstance[]>;
    list(params: SubscribedTrackListInstanceOptions, callback?: (error: Error | null, items: SubscribedTrackInstance[]) => any): Promise<SubscribedTrackInstance[]>;
    /**
     * Retrieve a single page of SubscribedTrackInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SubscribedTrackListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SubscribedTrackPage) => any): Promise<SubscribedTrackPage>;
    page(params: SubscribedTrackListInstancePageOptions, callback?: (error: Error | null, items: SubscribedTrackPage) => any): Promise<SubscribedTrackPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SubscribedTrackListInstance(version: V1, roomSid: string, participantSid: string): SubscribedTrackListInstance;
export declare class SubscribedTrackPage extends Page<V1, SubscribedTrackPayload, SubscribedTrackResource, SubscribedTrackInstance> {
    /**
     * Initialize the SubscribedTrackPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SubscribedTrackSolution);
    /**
     * Build an instance of SubscribedTrackInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SubscribedTrackResource): SubscribedTrackInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
