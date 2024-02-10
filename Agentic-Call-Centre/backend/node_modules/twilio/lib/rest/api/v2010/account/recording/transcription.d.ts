/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2010 from "../../../V2010";
export type TranscriptionStatus = "in-progress" | "completed" | "failed";
/**
 * Options to pass to each
 */
export interface TranscriptionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TranscriptionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TranscriptionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TranscriptionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TranscriptionContext {
    /**
     * Remove a TranscriptionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TranscriptionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TranscriptionInstance
     */
    fetch(callback?: (error: Error | null, item?: TranscriptionInstance) => any): Promise<TranscriptionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TranscriptionContextSolution {
    accountSid: string;
    recordingSid: string;
    sid: string;
}
export declare class TranscriptionContextImpl implements TranscriptionContext {
    protected _version: V2010;
    protected _solution: TranscriptionContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, recordingSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: TranscriptionInstance) => any): Promise<TranscriptionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TranscriptionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TranscriptionPayload extends TwilioResponsePayload {
    transcriptions: TranscriptionResource[];
}
interface TranscriptionResource {
    account_sid: string;
    api_version: string;
    date_created: Date;
    date_updated: Date;
    duration: string;
    price: number;
    price_unit: string;
    recording_sid: string;
    sid: string;
    status: TranscriptionStatus;
    transcription_text: string;
    type: string;
    uri: string;
}
export declare class TranscriptionInstance {
    protected _version: V2010;
    protected _solution: TranscriptionContextSolution;
    protected _context?: TranscriptionContext;
    constructor(_version: V2010, payload: TranscriptionResource, accountSid: string, recordingSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Transcription resource.
     */
    accountSid: string;
    /**
     * The API version used to create the transcription.
     */
    apiVersion: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The duration of the transcribed audio in seconds.
     */
    duration: string;
    /**
     * The charge for the transcript in the currency associated with the account. This value is populated after the transcript is complete so it may not be available immediately.
     */
    price: number;
    /**
     * The currency in which `price` is measured, in [ISO 4127](https://www.iso.org/iso/home/standards/currency_codes.htm) format (e.g. `usd`, `eur`, `jpy`).
     */
    priceUnit: string;
    /**
     * The SID of the [Recording](https://www.twilio.com/docs/voice/api/recording) from which the transcription was created.
     */
    recordingSid: string;
    /**
     * The unique string that that we created to identify the Transcription resource.
     */
    sid: string;
    status: TranscriptionStatus;
    /**
     * The text content of the transcription.
     */
    transcriptionText: string;
    /**
     * The transcription type.
     */
    type: string;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Remove a TranscriptionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TranscriptionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TranscriptionInstance
     */
    fetch(callback?: (error: Error | null, item?: TranscriptionInstance) => any): Promise<TranscriptionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        apiVersion: string;
        dateCreated: Date;
        dateUpdated: Date;
        duration: string;
        price: number;
        priceUnit: string;
        recordingSid: string;
        sid: string;
        status: TranscriptionStatus;
        transcriptionText: string;
        type: string;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TranscriptionSolution {
    accountSid: string;
    recordingSid: string;
}
export interface TranscriptionListInstance {
    _version: V2010;
    _solution: TranscriptionSolution;
    _uri: string;
    (sid: string): TranscriptionContext;
    get(sid: string): TranscriptionContext;
    /**
     * Streams TranscriptionInstance records from the API.
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
     * @param { TranscriptionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TranscriptionInstance, done: (err?: Error) => void) => void): void;
    each(params: TranscriptionListInstanceEachOptions, callback?: (item: TranscriptionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TranscriptionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TranscriptionPage) => any): Promise<TranscriptionPage>;
    /**
     * Lists TranscriptionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TranscriptionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TranscriptionInstance[]) => any): Promise<TranscriptionInstance[]>;
    list(params: TranscriptionListInstanceOptions, callback?: (error: Error | null, items: TranscriptionInstance[]) => any): Promise<TranscriptionInstance[]>;
    /**
     * Retrieve a single page of TranscriptionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TranscriptionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TranscriptionPage) => any): Promise<TranscriptionPage>;
    page(params: TranscriptionListInstancePageOptions, callback?: (error: Error | null, items: TranscriptionPage) => any): Promise<TranscriptionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TranscriptionListInstance(version: V2010, accountSid: string, recordingSid: string): TranscriptionListInstance;
export declare class TranscriptionPage extends Page<V2010, TranscriptionPayload, TranscriptionResource, TranscriptionInstance> {
    /**
     * Initialize the TranscriptionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: TranscriptionSolution);
    /**
     * Build an instance of TranscriptionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TranscriptionResource): TranscriptionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
