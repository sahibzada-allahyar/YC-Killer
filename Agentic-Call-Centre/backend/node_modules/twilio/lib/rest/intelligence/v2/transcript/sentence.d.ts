/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
/**
 * Options to pass to each
 */
export interface SentenceListInstanceEachOptions {
    /** Grant access to PII Redacted/Unredacted Sentences. If redaction is enabled, the default is `true` to access redacted sentences. */
    redacted?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SentenceInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SentenceListInstanceOptions {
    /** Grant access to PII Redacted/Unredacted Sentences. If redaction is enabled, the default is `true` to access redacted sentences. */
    redacted?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SentenceListInstancePageOptions {
    /** Grant access to PII Redacted/Unredacted Sentences. If redaction is enabled, the default is `true` to access redacted sentences. */
    redacted?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SentenceSolution {
    transcriptSid: string;
}
export interface SentenceListInstance {
    _version: V2;
    _solution: SentenceSolution;
    _uri: string;
    /**
     * Streams SentenceInstance records from the API.
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
     * @param { SentenceListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SentenceInstance, done: (err?: Error) => void) => void): void;
    each(params: SentenceListInstanceEachOptions, callback?: (item: SentenceInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SentenceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SentencePage) => any): Promise<SentencePage>;
    /**
     * Lists SentenceInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SentenceListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SentenceInstance[]) => any): Promise<SentenceInstance[]>;
    list(params: SentenceListInstanceOptions, callback?: (error: Error | null, items: SentenceInstance[]) => any): Promise<SentenceInstance[]>;
    /**
     * Retrieve a single page of SentenceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SentenceListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SentencePage) => any): Promise<SentencePage>;
    page(params: SentenceListInstancePageOptions, callback?: (error: Error | null, items: SentencePage) => any): Promise<SentencePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SentenceListInstance(version: V2, transcriptSid: string): SentenceListInstance;
interface SentencePayload extends TwilioResponsePayload {
    sentences: SentenceResource[];
}
interface SentenceResource {
    media_channel: number;
    sentence_index: number;
    start_time: number;
    end_time: number;
    transcript: string;
    sid: string;
    confidence: number;
}
export declare class SentenceInstance {
    protected _version: V2;
    constructor(_version: V2, payload: SentenceResource, transcriptSid: string);
    /**
     * The channel number.
     */
    mediaChannel: number;
    /**
     * The index of the sentence in the transcript.
     */
    sentenceIndex: number;
    /**
     * Offset from the beginning of the transcript when this sentence starts.
     */
    startTime: number;
    /**
     * Offset from the beginning of the transcript when this sentence ends.
     */
    endTime: number;
    /**
     * Transcript text.
     */
    transcript: string;
    /**
     * A 34 character string that uniquely identifies this Sentence.
     */
    sid: string;
    confidence: number;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        mediaChannel: number;
        sentenceIndex: number;
        startTime: number;
        endTime: number;
        transcript: string;
        sid: string;
        confidence: number;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class SentencePage extends Page<V2, SentencePayload, SentenceResource, SentenceInstance> {
    /**
     * Initialize the SentencePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: SentenceSolution);
    /**
     * Build an instance of SentenceInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SentenceResource): SentenceInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
