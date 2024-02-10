/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type CompositionHookFormat = "mp4" | "webm";
/**
 * Options to pass to update a CompositionHookInstance
 */
export interface CompositionHookContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to  100 characters long and it must be unique within the account. */
    friendlyName: string;
    /** Whether the composition hook is active. When `true`, the composition hook will be triggered for every completed Group Room in the account. When `false`, the composition hook never triggers. */
    enabled?: boolean;
    /** A JSON object that describes the video layout of the composition hook in terms of regions. See [Specifying Video Layouts](https://www.twilio.com/docs/video/api/compositions-resource#specifying-video-layouts) for more info. */
    videoLayout?: any;
    /** An array of track names from the same group room to merge into the compositions created by the composition hook. Can include zero or more track names. A composition triggered by the composition hook includes all audio sources specified in `audio_sources` except those specified in `audio_sources_excluded`. The track names in this parameter can include an asterisk as a wild card character, which matches zero or more characters in a track name. For example, `student*` includes tracks named `student` as well as `studentTeam`. */
    audioSources?: Array<string>;
    /** An array of track names to exclude. A composition triggered by the composition hook includes all audio sources specified in `audio_sources` except for those specified in `audio_sources_excluded`. The track names in this parameter can include an asterisk as a wild card character, which matches zero or more characters in a track name. For example, `student*` excludes `student` as well as `studentTeam`. This parameter can also be empty. */
    audioSourcesExcluded?: Array<string>;
    /** Whether to clip the intervals where there is no active media in the compositions triggered by the composition hook. The default is `true`. Compositions with `trim` enabled are shorter when the Room is created and no Participant joins for a while as well as if all the Participants leave the room and join later, because those gaps will be removed. See [Specifying Video Layouts](https://www.twilio.com/docs/video/api/compositions-resource#specifying-video-layouts) for more info. */
    trim?: boolean;
    /**  */
    format?: CompositionHookFormat;
    /** A string that describes the columns (width) and rows (height) of the generated composed video in pixels. Defaults to `640x480`.  The string\\\'s format is `{width}x{height}` where:   * 16 <= `{width}` <= 1280 * 16 <= `{height}` <= 1280 * `{width}` * `{height}` <= 921,600  Typical values are:   * HD = `1280x720` * PAL = `1024x576` * VGA = `640x480` * CIF = `320x240`  Note that the `resolution` imposes an aspect ratio to the resulting composition. When the original video tracks are constrained by the aspect ratio, they are scaled to fit. See [Specifying Video Layouts](https://www.twilio.com/docs/video/api/compositions-resource#specifying-video-layouts) for more info. */
    resolution?: string;
    /** The URL we should call using the `status_callback_method` to send status information to your application on every composition event. If not provided, status callback events will not be dispatched. */
    statusCallback?: string;
    /** The HTTP method we should use to call `status_callback`. Can be: `POST` or `GET` and the default is `POST`. */
    statusCallbackMethod?: string;
}
/**
 * Options to pass to create a CompositionHookInstance
 */
export interface CompositionHookListInstanceCreateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to  100 characters long and it must be unique within the account. */
    friendlyName: string;
    /** Whether the composition hook is active. When `true`, the composition hook will be triggered for every completed Group Room in the account. When `false`, the composition hook will never be triggered. */
    enabled?: boolean;
    /** An object that describes the video layout of the composition hook in terms of regions. See [Specifying Video Layouts](https://www.twilio.com/docs/video/api/compositions-resource#specifying-video-layouts) for more info. */
    videoLayout?: any;
    /** An array of track names from the same group room to merge into the compositions created by the composition hook. Can include zero or more track names. A composition triggered by the composition hook includes all audio sources specified in `audio_sources` except those specified in `audio_sources_excluded`. The track names in this parameter can include an asterisk as a wild card character, which matches zero or more characters in a track name. For example, `student*` includes tracks named `student` as well as `studentTeam`. */
    audioSources?: Array<string>;
    /** An array of track names to exclude. A composition triggered by the composition hook includes all audio sources specified in `audio_sources` except for those specified in `audio_sources_excluded`. The track names in this parameter can include an asterisk as a wild card character, which matches zero or more characters in a track name. For example, `student*` excludes `student` as well as `studentTeam`. This parameter can also be empty. */
    audioSourcesExcluded?: Array<string>;
    /** A string that describes the columns (width) and rows (height) of the generated composed video in pixels. Defaults to `640x480`.  The string\\\'s format is `{width}x{height}` where:   * 16 <= `{width}` <= 1280 * 16 <= `{height}` <= 1280 * `{width}` * `{height}` <= 921,600  Typical values are:   * HD = `1280x720` * PAL = `1024x576` * VGA = `640x480` * CIF = `320x240`  Note that the `resolution` imposes an aspect ratio to the resulting composition. When the original video tracks are constrained by the aspect ratio, they are scaled to fit. See [Specifying Video Layouts](https://www.twilio.com/docs/video/api/compositions-resource#specifying-video-layouts) for more info. */
    resolution?: string;
    /**  */
    format?: CompositionHookFormat;
    /** The URL we should call using the `status_callback_method` to send status information to your application on every composition event. If not provided, status callback events will not be dispatched. */
    statusCallback?: string;
    /** The HTTP method we should use to call `status_callback`. Can be: `POST` or `GET` and the default is `POST`. */
    statusCallbackMethod?: string;
    /** Whether to clip the intervals where there is no active media in the Compositions triggered by the composition hook. The default is `true`. Compositions with `trim` enabled are shorter when the Room is created and no Participant joins for a while as well as if all the Participants leave the room and join later, because those gaps will be removed. See [Specifying Video Layouts](https://www.twilio.com/docs/video/api/compositions-resource#specifying-video-layouts) for more info. */
    trim?: boolean;
}
/**
 * Options to pass to each
 */
export interface CompositionHookListInstanceEachOptions {
    /** Read only CompositionHook resources with an `enabled` value that matches this parameter. */
    enabled?: boolean;
    /** Read only CompositionHook resources created on or after this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime with time zone. */
    dateCreatedAfter?: Date;
    /** Read only CompositionHook resources created before this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime with time zone. */
    dateCreatedBefore?: Date;
    /** Read only CompositionHook resources with friendly names that match this string. The match is not case sensitive and can include asterisk `*` characters as wildcard match. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: CompositionHookInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface CompositionHookListInstanceOptions {
    /** Read only CompositionHook resources with an `enabled` value that matches this parameter. */
    enabled?: boolean;
    /** Read only CompositionHook resources created on or after this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime with time zone. */
    dateCreatedAfter?: Date;
    /** Read only CompositionHook resources created before this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime with time zone. */
    dateCreatedBefore?: Date;
    /** Read only CompositionHook resources with friendly names that match this string. The match is not case sensitive and can include asterisk `*` characters as wildcard match. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CompositionHookListInstancePageOptions {
    /** Read only CompositionHook resources with an `enabled` value that matches this parameter. */
    enabled?: boolean;
    /** Read only CompositionHook resources created on or after this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime with time zone. */
    dateCreatedAfter?: Date;
    /** Read only CompositionHook resources created before this [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) datetime with time zone. */
    dateCreatedBefore?: Date;
    /** Read only CompositionHook resources with friendly names that match this string. The match is not case sensitive and can include asterisk `*` characters as wildcard match. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CompositionHookContext {
    /**
     * Remove a CompositionHookInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CompositionHookInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CompositionHookInstance
     */
    fetch(callback?: (error: Error | null, item?: CompositionHookInstance) => any): Promise<CompositionHookInstance>;
    /**
     * Update a CompositionHookInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CompositionHookInstance
     */
    update(params: CompositionHookContextUpdateOptions, callback?: (error: Error | null, item?: CompositionHookInstance) => any): Promise<CompositionHookInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface CompositionHookContextSolution {
    sid: string;
}
export declare class CompositionHookContextImpl implements CompositionHookContext {
    protected _version: V1;
    protected _solution: CompositionHookContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: CompositionHookInstance) => any): Promise<CompositionHookInstance>;
    update(params: CompositionHookContextUpdateOptions, callback?: (error: Error | null, item?: CompositionHookInstance) => any): Promise<CompositionHookInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): CompositionHookContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface CompositionHookPayload extends TwilioResponsePayload {
    composition_hooks: CompositionHookResource[];
}
interface CompositionHookResource {
    account_sid: string;
    friendly_name: string;
    enabled: boolean;
    date_created: Date;
    date_updated: Date;
    sid: string;
    audio_sources: Array<string>;
    audio_sources_excluded: Array<string>;
    video_layout: any;
    resolution: string;
    trim: boolean;
    format: CompositionHookFormat;
    status_callback: string;
    status_callback_method: string;
    url: string;
}
export declare class CompositionHookInstance {
    protected _version: V1;
    protected _solution: CompositionHookContextSolution;
    protected _context?: CompositionHookContext;
    constructor(_version: V1, payload: CompositionHookResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the CompositionHook resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the resource. Can be up to 100 characters long and must be unique within the account.
     */
    friendlyName: string;
    /**
     * Whether the CompositionHook is active. When `true`, the CompositionHook is triggered for every completed Group Room on the account. When `false`, the CompositionHook is never triggered.
     */
    enabled: boolean;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The unique string that we created to identify the CompositionHook resource.
     */
    sid: string;
    /**
     * The array of track names to include in the compositions created by the composition hook. A composition triggered by the composition hook includes all audio sources specified in `audio_sources` except those specified in `audio_sources_excluded`. The track names in this property can include an asterisk as a wild card character, which matches zero or more characters in a track name. For example, `student*` includes tracks named `student` as well as `studentTeam`. Please, be aware that either video_layout or audio_sources have to be provided to get a valid creation request
     */
    audioSources: Array<string>;
    /**
     * The array of track names to exclude from the compositions created by the composition hook. A composition triggered by the composition hook includes all audio sources specified in `audio_sources` except for those specified in `audio_sources_excluded`. The track names in this property can include an asterisk as a wild card character, which matches zero or more characters in a track name. For example, `student*` excludes `student` as well as `studentTeam`. This parameter can also be empty.
     */
    audioSourcesExcluded: Array<string>;
    /**
     * A JSON object that describes the video layout of the composition in terms of regions as specified in the HTTP POST request that created the CompositionHook resource. See [POST Parameters](https://www.twilio.com/docs/video/api/compositions-resource#http-post-parameters) for more information. Please, be aware that either video_layout or audio_sources have to be provided to get a valid creation request
     */
    videoLayout: any;
    /**
     * The dimensions of the video image in pixels expressed as columns (width) and rows (height). The string\'s format is `{width}x{height}`, such as `640x480`.
     */
    resolution: string;
    /**
     * Whether intervals with no media are clipped, as specified in the POST request that created the CompositionHook resource. Compositions with `trim` enabled are shorter when the Room is created and no Participant joins for a while as well as if all the Participants leave the room and join later, because those gaps will be removed. See [Specifying Video Layouts](https://www.twilio.com/docs/video/api/compositions-resource#specifying-video-layouts) for more info.
     */
    trim: boolean;
    format: CompositionHookFormat;
    /**
     * The URL we call using the `status_callback_method` to send status information to your application.
     */
    statusCallback: string;
    /**
     * The HTTP method we should use to call `status_callback`. Can be `POST` or `GET` and defaults to `POST`.
     */
    statusCallbackMethod: string;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a CompositionHookInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CompositionHookInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CompositionHookInstance
     */
    fetch(callback?: (error: Error | null, item?: CompositionHookInstance) => any): Promise<CompositionHookInstance>;
    /**
     * Update a CompositionHookInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CompositionHookInstance
     */
    update(params: CompositionHookContextUpdateOptions, callback?: (error: Error | null, item?: CompositionHookInstance) => any): Promise<CompositionHookInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        friendlyName: string;
        enabled: boolean;
        dateCreated: Date;
        dateUpdated: Date;
        sid: string;
        audioSources: string[];
        audioSourcesExcluded: string[];
        videoLayout: any;
        resolution: string;
        trim: boolean;
        format: CompositionHookFormat;
        statusCallback: string;
        statusCallbackMethod: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CompositionHookSolution {
}
export interface CompositionHookListInstance {
    _version: V1;
    _solution: CompositionHookSolution;
    _uri: string;
    (sid: string): CompositionHookContext;
    get(sid: string): CompositionHookContext;
    /**
     * Create a CompositionHookInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CompositionHookInstance
     */
    create(params: CompositionHookListInstanceCreateOptions, callback?: (error: Error | null, item?: CompositionHookInstance) => any): Promise<CompositionHookInstance>;
    /**
     * Streams CompositionHookInstance records from the API.
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
     * @param { CompositionHookListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: CompositionHookInstance, done: (err?: Error) => void) => void): void;
    each(params: CompositionHookListInstanceEachOptions, callback?: (item: CompositionHookInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of CompositionHookInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: CompositionHookPage) => any): Promise<CompositionHookPage>;
    /**
     * Lists CompositionHookInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CompositionHookListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: CompositionHookInstance[]) => any): Promise<CompositionHookInstance[]>;
    list(params: CompositionHookListInstanceOptions, callback?: (error: Error | null, items: CompositionHookInstance[]) => any): Promise<CompositionHookInstance[]>;
    /**
     * Retrieve a single page of CompositionHookInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CompositionHookListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: CompositionHookPage) => any): Promise<CompositionHookPage>;
    page(params: CompositionHookListInstancePageOptions, callback?: (error: Error | null, items: CompositionHookPage) => any): Promise<CompositionHookPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CompositionHookListInstance(version: V1): CompositionHookListInstance;
export declare class CompositionHookPage extends Page<V1, CompositionHookPayload, CompositionHookResource, CompositionHookInstance> {
    /**
     * Initialize the CompositionHookPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: CompositionHookSolution);
    /**
     * Build an instance of CompositionHookInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CompositionHookResource): CompositionHookInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
