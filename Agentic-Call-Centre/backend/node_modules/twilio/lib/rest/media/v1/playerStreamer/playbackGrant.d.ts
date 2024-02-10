/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
/**
 * Options to pass to create a PlaybackGrantInstance
 */
export interface PlaybackGrantContextCreateOptions {
    /** The time to live of the PlaybackGrant. Default value is 15 seconds. Maximum value is 60 seconds. */
    ttl?: number;
    /** The full origin URL where the livestream can be streamed. If this is not provided, it can be streamed from any domain. */
    accessControlAllowOrigin?: string;
}
export interface PlaybackGrantContext {
    /**
     * Create a PlaybackGrantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PlaybackGrantInstance
     */
    create(callback?: (error: Error | null, item?: PlaybackGrantInstance) => any): Promise<PlaybackGrantInstance>;
    /**
     * Create a PlaybackGrantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PlaybackGrantInstance
     */
    create(params: PlaybackGrantContextCreateOptions, callback?: (error: Error | null, item?: PlaybackGrantInstance) => any): Promise<PlaybackGrantInstance>;
    /**
     * Fetch a PlaybackGrantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PlaybackGrantInstance
     */
    fetch(callback?: (error: Error | null, item?: PlaybackGrantInstance) => any): Promise<PlaybackGrantInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PlaybackGrantContextSolution {
    sid: string;
}
export declare class PlaybackGrantContextImpl implements PlaybackGrantContext {
    protected _version: V1;
    protected _solution: PlaybackGrantContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    create(params?: PlaybackGrantContextCreateOptions | ((error: Error | null, item?: PlaybackGrantInstance) => any), callback?: (error: Error | null, item?: PlaybackGrantInstance) => any): Promise<PlaybackGrantInstance>;
    fetch(callback?: (error: Error | null, item?: PlaybackGrantInstance) => any): Promise<PlaybackGrantInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PlaybackGrantContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PlaybackGrantResource {
    sid: string;
    url: string;
    account_sid: string;
    date_created: Date;
    grant: any;
}
export declare class PlaybackGrantInstance {
    protected _version: V1;
    protected _solution: PlaybackGrantContextSolution;
    protected _context?: PlaybackGrantContext;
    constructor(_version: V1, payload: PlaybackGrantResource, sid: string);
    /**
     * The unique string generated to identify the PlayerStreamer resource that this PlaybackGrant authorizes views for.
     */
    sid: string;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created this resource.
     */
    accountSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The grant that authorizes the player sdk to connect to the livestream
     */
    grant: any;
    private get _proxy();
    /**
     * Create a PlaybackGrantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PlaybackGrantInstance
     */
    create(callback?: (error: Error | null, item?: PlaybackGrantInstance) => any): Promise<PlaybackGrantInstance>;
    /**
     * Create a PlaybackGrantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PlaybackGrantInstance
     */
    create(params: PlaybackGrantContextCreateOptions, callback?: (error: Error | null, item?: PlaybackGrantInstance) => any): Promise<PlaybackGrantInstance>;
    /**
     * Fetch a PlaybackGrantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PlaybackGrantInstance
     */
    fetch(callback?: (error: Error | null, item?: PlaybackGrantInstance) => any): Promise<PlaybackGrantInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        url: string;
        accountSid: string;
        dateCreated: Date;
        grant: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PlaybackGrantSolution {
    sid: string;
}
export interface PlaybackGrantListInstance {
    _version: V1;
    _solution: PlaybackGrantSolution;
    _uri: string;
    (): PlaybackGrantContext;
    get(): PlaybackGrantContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PlaybackGrantListInstance(version: V1, sid: string): PlaybackGrantListInstance;
export {};
