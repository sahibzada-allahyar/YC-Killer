/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../../V2";
/**
 * Options to pass to fetch a MediaInstance
 */
export interface MediaContextFetchOptions {
    /** Grant access to PII Redacted/Unredacted Media. If redaction is enabled, the default is `true` to access redacted media. */
    redacted?: boolean;
}
export interface MediaContext {
    /**
     * Fetch a MediaInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaInstance
     */
    fetch(callback?: (error: Error | null, item?: MediaInstance) => any): Promise<MediaInstance>;
    /**
     * Fetch a MediaInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaInstance
     */
    fetch(params: MediaContextFetchOptions, callback?: (error: Error | null, item?: MediaInstance) => any): Promise<MediaInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface MediaContextSolution {
    sid: string;
}
export declare class MediaContextImpl implements MediaContext {
    protected _version: V2;
    protected _solution: MediaContextSolution;
    protected _uri: string;
    constructor(_version: V2, sid: string);
    fetch(params?: MediaContextFetchOptions | ((error: Error | null, item?: MediaInstance) => any), callback?: (error: Error | null, item?: MediaInstance) => any): Promise<MediaInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): MediaContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface MediaResource {
    account_sid: string;
    media_url: string;
    service_sid: string;
    sid: string;
    url: string;
}
export declare class MediaInstance {
    protected _version: V2;
    protected _solution: MediaContextSolution;
    protected _context?: MediaContext;
    constructor(_version: V2, payload: MediaResource, sid: string);
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * Downloadable URL for media, if stored in Twilio AI.
     */
    mediaUrl: string;
    /**
     * The unique SID identifier of the Service.
     */
    serviceSid: string;
    /**
     * The unique SID identifier of the Transcript.
     */
    sid: string;
    /**
     * The URL of this resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a MediaInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaInstance
     */
    fetch(callback?: (error: Error | null, item?: MediaInstance) => any): Promise<MediaInstance>;
    /**
     * Fetch a MediaInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaInstance
     */
    fetch(params: MediaContextFetchOptions, callback?: (error: Error | null, item?: MediaInstance) => any): Promise<MediaInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        mediaUrl: string;
        serviceSid: string;
        sid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface MediaSolution {
    sid: string;
}
export interface MediaListInstance {
    _version: V2;
    _solution: MediaSolution;
    _uri: string;
    (): MediaContext;
    get(): MediaContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function MediaListInstance(version: V2, sid: string): MediaListInstance;
export {};
