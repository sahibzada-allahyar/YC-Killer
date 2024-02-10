/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2010 from "../../../V2010";
/**
 * Options to pass to each
 */
export interface MediaListInstanceEachOptions {
    /** Only include Media resources that were created on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read Media that were created on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read Media that were created on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read Media that were created on or after midnight of this date. */
    dateCreated?: Date;
    /** Only include Media resources that were created on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read Media that were created on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read Media that were created on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read Media that were created on or after midnight of this date. */
    dateCreatedBefore?: Date;
    /** Only include Media resources that were created on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read Media that were created on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read Media that were created on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read Media that were created on or after midnight of this date. */
    dateCreatedAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: MediaInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface MediaListInstanceOptions {
    /** Only include Media resources that were created on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read Media that were created on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read Media that were created on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read Media that were created on or after midnight of this date. */
    dateCreated?: Date;
    /** Only include Media resources that were created on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read Media that were created on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read Media that were created on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read Media that were created on or after midnight of this date. */
    dateCreatedBefore?: Date;
    /** Only include Media resources that were created on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read Media that were created on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read Media that were created on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read Media that were created on or after midnight of this date. */
    dateCreatedAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface MediaListInstancePageOptions {
    /** Only include Media resources that were created on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read Media that were created on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read Media that were created on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read Media that were created on or after midnight of this date. */
    dateCreated?: Date;
    /** Only include Media resources that were created on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read Media that were created on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read Media that were created on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read Media that were created on or after midnight of this date. */
    dateCreatedBefore?: Date;
    /** Only include Media resources that were created on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read Media that were created on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read Media that were created on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read Media that were created on or after midnight of this date. */
    dateCreatedAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface MediaContext {
    /**
     * Remove a MediaInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a MediaInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaInstance
     */
    fetch(callback?: (error: Error | null, item?: MediaInstance) => any): Promise<MediaInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface MediaContextSolution {
    accountSid: string;
    messageSid: string;
    sid: string;
}
export declare class MediaContextImpl implements MediaContext {
    protected _version: V2010;
    protected _solution: MediaContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, messageSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: MediaInstance) => any): Promise<MediaInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): MediaContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface MediaPayload extends TwilioResponsePayload {
    media_list: MediaResource[];
}
interface MediaResource {
    account_sid: string;
    content_type: string;
    date_created: Date;
    date_updated: Date;
    parent_sid: string;
    sid: string;
    uri: string;
}
export declare class MediaInstance {
    protected _version: V2010;
    protected _solution: MediaContextSolution;
    protected _context?: MediaContext;
    constructor(_version: V2010, payload: MediaResource, accountSid: string, messageSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) associated with this Media resource.
     */
    accountSid: string;
    /**
     * The default [MIME type](https://en.wikipedia.org/wiki/Internet_media_type) of the media, for example `image/jpeg`, `image/png`, or `image/gif`.
     */
    contentType: string;
    /**
     * The date and time in GMT when this Media resource was created, specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when this Media resource was last updated, specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The SID of the Message resource that is associated with this Media resource.
     */
    parentSid: string;
    /**
     * The unique string that identifies this Media resource.
     */
    sid: string;
    /**
     * The URI of this Media resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Remove a MediaInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a MediaInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MediaInstance
     */
    fetch(callback?: (error: Error | null, item?: MediaInstance) => any): Promise<MediaInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        contentType: string;
        dateCreated: Date;
        dateUpdated: Date;
        parentSid: string;
        sid: string;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface MediaSolution {
    accountSid: string;
    messageSid: string;
}
export interface MediaListInstance {
    _version: V2010;
    _solution: MediaSolution;
    _uri: string;
    (sid: string): MediaContext;
    get(sid: string): MediaContext;
    /**
     * Streams MediaInstance records from the API.
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
     * @param { MediaListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: MediaInstance, done: (err?: Error) => void) => void): void;
    each(params: MediaListInstanceEachOptions, callback?: (item: MediaInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of MediaInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: MediaPage) => any): Promise<MediaPage>;
    /**
     * Lists MediaInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MediaListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: MediaInstance[]) => any): Promise<MediaInstance[]>;
    list(params: MediaListInstanceOptions, callback?: (error: Error | null, items: MediaInstance[]) => any): Promise<MediaInstance[]>;
    /**
     * Retrieve a single page of MediaInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MediaListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: MediaPage) => any): Promise<MediaPage>;
    page(params: MediaListInstancePageOptions, callback?: (error: Error | null, items: MediaPage) => any): Promise<MediaPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function MediaListInstance(version: V2010, accountSid: string, messageSid: string): MediaListInstance;
export declare class MediaPage extends Page<V2010, MediaPayload, MediaResource, MediaInstance> {
    /**
     * Initialize the MediaPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: MediaSolution);
    /**
     * Build an instance of MediaInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: MediaResource): MediaInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
