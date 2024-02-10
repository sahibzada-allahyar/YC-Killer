/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
export type PublishedTrackKind = "audio" | "video" | "data";
/**
 * Options to pass to each
 */
export interface PublishedTrackListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: PublishedTrackInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface PublishedTrackListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface PublishedTrackListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface PublishedTrackContext {
    /**
     * Fetch a PublishedTrackInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PublishedTrackInstance
     */
    fetch(callback?: (error: Error | null, item?: PublishedTrackInstance) => any): Promise<PublishedTrackInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PublishedTrackContextSolution {
    roomSid: string;
    participantSid: string;
    sid: string;
}
export declare class PublishedTrackContextImpl implements PublishedTrackContext {
    protected _version: V1;
    protected _solution: PublishedTrackContextSolution;
    protected _uri: string;
    constructor(_version: V1, roomSid: string, participantSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: PublishedTrackInstance) => any): Promise<PublishedTrackInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PublishedTrackContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PublishedTrackPayload extends TwilioResponsePayload {
    published_tracks: PublishedTrackResource[];
}
interface PublishedTrackResource {
    sid: string;
    participant_sid: string;
    room_sid: string;
    name: string;
    date_created: Date;
    date_updated: Date;
    enabled: boolean;
    kind: PublishedTrackKind;
    url: string;
}
export declare class PublishedTrackInstance {
    protected _version: V1;
    protected _solution: PublishedTrackContextSolution;
    protected _context?: PublishedTrackContext;
    constructor(_version: V1, payload: PublishedTrackResource, roomSid: string, participantSid: string, sid?: string);
    /**
     * The unique string that we created to identify the RoomParticipantPublishedTrack resource.
     */
    sid: string;
    /**
     * The SID of the Participant resource with the published track.
     */
    participantSid: string;
    /**
     * The SID of the Room resource where the track is published.
     */
    roomSid: string;
    /**
     * The track name. Must be no more than 128 characters, and be unique among the participant\'s published tracks.
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
    kind: PublishedTrackKind;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a PublishedTrackInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PublishedTrackInstance
     */
    fetch(callback?: (error: Error | null, item?: PublishedTrackInstance) => any): Promise<PublishedTrackInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        participantSid: string;
        roomSid: string;
        name: string;
        dateCreated: Date;
        dateUpdated: Date;
        enabled: boolean;
        kind: PublishedTrackKind;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PublishedTrackSolution {
    roomSid: string;
    participantSid: string;
}
export interface PublishedTrackListInstance {
    _version: V1;
    _solution: PublishedTrackSolution;
    _uri: string;
    (sid: string): PublishedTrackContext;
    get(sid: string): PublishedTrackContext;
    /**
     * Streams PublishedTrackInstance records from the API.
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
     * @param { PublishedTrackListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: PublishedTrackInstance, done: (err?: Error) => void) => void): void;
    each(params: PublishedTrackListInstanceEachOptions, callback?: (item: PublishedTrackInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of PublishedTrackInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: PublishedTrackPage) => any): Promise<PublishedTrackPage>;
    /**
     * Lists PublishedTrackInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PublishedTrackListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: PublishedTrackInstance[]) => any): Promise<PublishedTrackInstance[]>;
    list(params: PublishedTrackListInstanceOptions, callback?: (error: Error | null, items: PublishedTrackInstance[]) => any): Promise<PublishedTrackInstance[]>;
    /**
     * Retrieve a single page of PublishedTrackInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PublishedTrackListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: PublishedTrackPage) => any): Promise<PublishedTrackPage>;
    page(params: PublishedTrackListInstancePageOptions, callback?: (error: Error | null, items: PublishedTrackPage) => any): Promise<PublishedTrackPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PublishedTrackListInstance(version: V1, roomSid: string, participantSid: string): PublishedTrackListInstance;
export declare class PublishedTrackPage extends Page<V1, PublishedTrackPayload, PublishedTrackResource, PublishedTrackInstance> {
    /**
     * Initialize the PublishedTrackPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: PublishedTrackSolution);
    /**
     * Build an instance of PublishedTrackInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: PublishedTrackResource): PublishedTrackInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
