/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V2 from "../V2";
import { MediaListInstance } from "./transcript/media";
import { OperatorResultListInstance } from "./transcript/operatorResult";
import { SentenceListInstance } from "./transcript/sentence";
export type TranscriptStatus = "queued" | "in-progress" | "completed" | "failed" | "canceled";
/**
 * Options to pass to create a TranscriptInstance
 */
export interface TranscriptListInstanceCreateOptions {
    /** The unique SID identifier of the Service. */
    serviceSid: string;
    /** JSON object describing Media Channel including Source and Participants */
    channel: any;
    /** Used to store client provided metadata. Maximum of 64 double-byte UTF8 characters. */
    customerKey?: string;
    /** The date that this Transcript\\\'s media was started, given in ISO 8601 format. */
    mediaStartTime?: Date;
}
/**
 * Options to pass to each
 */
export interface TranscriptListInstanceEachOptions {
    /** The unique SID identifier of the Service. */
    serviceSid?: string;
    /** Filter by before StartTime. */
    beforeStartTime?: string;
    /** Filter by after StartTime. */
    afterStartTime?: string;
    /** Filter by before DateCreated. */
    beforeDateCreated?: string;
    /** Filter by after DateCreated. */
    afterDateCreated?: string;
    /** Filter by status. */
    status?: string;
    /** Filter by Language Code. */
    languageCode?: string;
    /** Filter by SourceSid. */
    sourceSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TranscriptInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TranscriptListInstanceOptions {
    /** The unique SID identifier of the Service. */
    serviceSid?: string;
    /** Filter by before StartTime. */
    beforeStartTime?: string;
    /** Filter by after StartTime. */
    afterStartTime?: string;
    /** Filter by before DateCreated. */
    beforeDateCreated?: string;
    /** Filter by after DateCreated. */
    afterDateCreated?: string;
    /** Filter by status. */
    status?: string;
    /** Filter by Language Code. */
    languageCode?: string;
    /** Filter by SourceSid. */
    sourceSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TranscriptListInstancePageOptions {
    /** The unique SID identifier of the Service. */
    serviceSid?: string;
    /** Filter by before StartTime. */
    beforeStartTime?: string;
    /** Filter by after StartTime. */
    afterStartTime?: string;
    /** Filter by before DateCreated. */
    beforeDateCreated?: string;
    /** Filter by after DateCreated. */
    afterDateCreated?: string;
    /** Filter by status. */
    status?: string;
    /** Filter by Language Code. */
    languageCode?: string;
    /** Filter by SourceSid. */
    sourceSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TranscriptContext {
    media: MediaListInstance;
    operatorResults: OperatorResultListInstance;
    sentences: SentenceListInstance;
    /**
     * Remove a TranscriptInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TranscriptInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TranscriptInstance
     */
    fetch(callback?: (error: Error | null, item?: TranscriptInstance) => any): Promise<TranscriptInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TranscriptContextSolution {
    sid: string;
}
export declare class TranscriptContextImpl implements TranscriptContext {
    protected _version: V2;
    protected _solution: TranscriptContextSolution;
    protected _uri: string;
    protected _media?: MediaListInstance;
    protected _operatorResults?: OperatorResultListInstance;
    protected _sentences?: SentenceListInstance;
    constructor(_version: V2, sid: string);
    get media(): MediaListInstance;
    get operatorResults(): OperatorResultListInstance;
    get sentences(): SentenceListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: TranscriptInstance) => any): Promise<TranscriptInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TranscriptContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TranscriptPayload extends TwilioResponsePayload {
    transcripts: TranscriptResource[];
}
interface TranscriptResource {
    account_sid: string;
    service_sid: string;
    sid: string;
    date_created: Date;
    date_updated: Date;
    status: TranscriptStatus;
    channel: any;
    data_logging: boolean;
    language_code: string;
    customer_key: string;
    media_start_time: Date;
    duration: number;
    url: string;
    redaction: boolean;
    links: Record<string, string>;
}
export declare class TranscriptInstance {
    protected _version: V2;
    protected _solution: TranscriptContextSolution;
    protected _context?: TranscriptContext;
    constructor(_version: V2, payload: TranscriptResource, sid?: string);
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * The unique SID identifier of the Service.
     */
    serviceSid: string;
    /**
     * A 34 character string that uniquely identifies this Transcript.
     */
    sid: string;
    /**
     * The date that this Transcript was created, given in ISO 8601 format.
     */
    dateCreated: Date;
    /**
     * The date that this Transcript was updated, given in ISO 8601 format.
     */
    dateUpdated: Date;
    status: TranscriptStatus;
    /**
     * Media Channel describing Transcript Source and Participant Mapping
     */
    channel: any;
    /**
     * Data logging allows Twilio to improve the quality of the speech recognition & language understanding services through using customer data to refine, fine tune and evaluate machine learning models. Note: Data logging cannot be activated via API, only via www.twilio.com, as it requires additional consent.
     */
    dataLogging: boolean;
    /**
     * The default language code of the audio.
     */
    languageCode: string;
    customerKey: string;
    /**
     * The date that this Transcript\'s media was started, given in ISO 8601 format.
     */
    mediaStartTime: Date;
    /**
     * The duration of this Transcript\'s source
     */
    duration: number;
    /**
     * The URL of this resource.
     */
    url: string;
    /**
     * If the transcript has been redacted, a redacted alternative of the transcript will be available.
     */
    redaction: boolean;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a TranscriptInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TranscriptInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TranscriptInstance
     */
    fetch(callback?: (error: Error | null, item?: TranscriptInstance) => any): Promise<TranscriptInstance>;
    /**
     * Access the media.
     */
    media(): MediaListInstance;
    /**
     * Access the operatorResults.
     */
    operatorResults(): OperatorResultListInstance;
    /**
     * Access the sentences.
     */
    sentences(): SentenceListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        serviceSid: string;
        sid: string;
        dateCreated: Date;
        dateUpdated: Date;
        status: TranscriptStatus;
        channel: any;
        dataLogging: boolean;
        languageCode: string;
        customerKey: string;
        mediaStartTime: Date;
        duration: number;
        url: string;
        redaction: boolean;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TranscriptSolution {
}
export interface TranscriptListInstance {
    _version: V2;
    _solution: TranscriptSolution;
    _uri: string;
    (sid: string): TranscriptContext;
    get(sid: string): TranscriptContext;
    /**
     * Create a TranscriptInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TranscriptInstance
     */
    create(params: TranscriptListInstanceCreateOptions, callback?: (error: Error | null, item?: TranscriptInstance) => any): Promise<TranscriptInstance>;
    /**
     * Streams TranscriptInstance records from the API.
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
     * @param { TranscriptListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TranscriptInstance, done: (err?: Error) => void) => void): void;
    each(params: TranscriptListInstanceEachOptions, callback?: (item: TranscriptInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TranscriptInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TranscriptPage) => any): Promise<TranscriptPage>;
    /**
     * Lists TranscriptInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TranscriptListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TranscriptInstance[]) => any): Promise<TranscriptInstance[]>;
    list(params: TranscriptListInstanceOptions, callback?: (error: Error | null, items: TranscriptInstance[]) => any): Promise<TranscriptInstance[]>;
    /**
     * Retrieve a single page of TranscriptInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TranscriptListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TranscriptPage) => any): Promise<TranscriptPage>;
    page(params: TranscriptListInstancePageOptions, callback?: (error: Error | null, items: TranscriptPage) => any): Promise<TranscriptPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TranscriptListInstance(version: V2): TranscriptListInstance;
export declare class TranscriptPage extends Page<V2, TranscriptPayload, TranscriptResource, TranscriptInstance> {
    /**
     * Initialize the TranscriptPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: TranscriptSolution);
    /**
     * Build an instance of TranscriptInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TranscriptResource): TranscriptInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
