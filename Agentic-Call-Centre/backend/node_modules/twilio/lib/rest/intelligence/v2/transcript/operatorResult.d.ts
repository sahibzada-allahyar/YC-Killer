/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
export type OperatorResultOperatorType = "conversation_classify" | "utterance_classify" | "extract" | "extract_normalize" | "pii_extract";
/**
 * Options to pass to fetch a OperatorResultInstance
 */
export interface OperatorResultContextFetchOptions {
    /** Grant access to PII redacted/unredacted Language Understanding operator. If redaction is enabled, the default is True. */
    redacted?: boolean;
}
/**
 * Options to pass to each
 */
export interface OperatorResultListInstanceEachOptions {
    /** Grant access to PII redacted/unredacted Language Understanding operator. If redaction is enabled, the default is True. */
    redacted?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: OperatorResultInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface OperatorResultListInstanceOptions {
    /** Grant access to PII redacted/unredacted Language Understanding operator. If redaction is enabled, the default is True. */
    redacted?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface OperatorResultListInstancePageOptions {
    /** Grant access to PII redacted/unredacted Language Understanding operator. If redaction is enabled, the default is True. */
    redacted?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface OperatorResultContext {
    /**
     * Fetch a OperatorResultInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OperatorResultInstance
     */
    fetch(callback?: (error: Error | null, item?: OperatorResultInstance) => any): Promise<OperatorResultInstance>;
    /**
     * Fetch a OperatorResultInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OperatorResultInstance
     */
    fetch(params: OperatorResultContextFetchOptions, callback?: (error: Error | null, item?: OperatorResultInstance) => any): Promise<OperatorResultInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface OperatorResultContextSolution {
    transcriptSid: string;
    operatorSid: string;
}
export declare class OperatorResultContextImpl implements OperatorResultContext {
    protected _version: V2;
    protected _solution: OperatorResultContextSolution;
    protected _uri: string;
    constructor(_version: V2, transcriptSid: string, operatorSid: string);
    fetch(params?: OperatorResultContextFetchOptions | ((error: Error | null, item?: OperatorResultInstance) => any), callback?: (error: Error | null, item?: OperatorResultInstance) => any): Promise<OperatorResultInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): OperatorResultContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface OperatorResultPayload extends TwilioResponsePayload {
    operator_results: OperatorResultResource[];
}
interface OperatorResultResource {
    operator_type: OperatorResultOperatorType;
    name: string;
    operator_sid: string;
    extract_match: boolean;
    match_probability: number;
    normalized_result: string;
    utterance_results: Array<any>;
    utterance_match: boolean;
    predicted_label: string;
    predicted_probability: number;
    label_probabilities: any;
    extract_results: any;
    text_generation_results: any;
    transcript_sid: string;
    url: string;
}
export declare class OperatorResultInstance {
    protected _version: V2;
    protected _solution: OperatorResultContextSolution;
    protected _context?: OperatorResultContext;
    constructor(_version: V2, payload: OperatorResultResource, transcriptSid: string, operatorSid?: string);
    operatorType: OperatorResultOperatorType;
    /**
     * The name of the applied Language Understanding.
     */
    name: string;
    /**
     * A 34 character string that identifies this Language Understanding operator sid.
     */
    operatorSid: string;
    /**
     * Boolean to tell if extract Language Understanding Processing model matches results.
     */
    extractMatch: boolean;
    /**
     * Percentage of \'matching\' class needed to consider a sentence matches
     */
    matchProbability: number;
    /**
     * Normalized output of extraction stage which matches Label.
     */
    normalizedResult: string;
    /**
     * List of mapped utterance object which matches sentences.
     */
    utteranceResults: Array<any>;
    /**
     * Boolean to tell if Utterance matches results.
     */
    utteranceMatch: boolean;
    /**
     * The \'matching\' class. This might be available on conversation classify model outputs.
     */
    predictedLabel: string;
    /**
     * Percentage of \'matching\' class needed to consider a sentence matches.
     */
    predictedProbability: number;
    /**
     * The labels probabilities. This might be available on conversation classify model outputs.
     */
    labelProbabilities: any;
    /**
     * List of text extraction results. This might be available on classify-extract model outputs.
     */
    extractResults: any;
    /**
     * Output of a text generation operator for example Conversation Sumamary.
     */
    textGenerationResults: any;
    /**
     * A 34 character string that uniquely identifies this Transcript.
     */
    transcriptSid: string;
    /**
     * The URL of this resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a OperatorResultInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OperatorResultInstance
     */
    fetch(callback?: (error: Error | null, item?: OperatorResultInstance) => any): Promise<OperatorResultInstance>;
    /**
     * Fetch a OperatorResultInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OperatorResultInstance
     */
    fetch(params: OperatorResultContextFetchOptions, callback?: (error: Error | null, item?: OperatorResultInstance) => any): Promise<OperatorResultInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        operatorType: OperatorResultOperatorType;
        name: string;
        operatorSid: string;
        extractMatch: boolean;
        matchProbability: number;
        normalizedResult: string;
        utteranceResults: any[];
        utteranceMatch: boolean;
        predictedLabel: string;
        predictedProbability: number;
        labelProbabilities: any;
        extractResults: any;
        textGenerationResults: any;
        transcriptSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface OperatorResultSolution {
    transcriptSid: string;
}
export interface OperatorResultListInstance {
    _version: V2;
    _solution: OperatorResultSolution;
    _uri: string;
    (operatorSid: string): OperatorResultContext;
    get(operatorSid: string): OperatorResultContext;
    /**
     * Streams OperatorResultInstance records from the API.
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
     * @param { OperatorResultListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: OperatorResultInstance, done: (err?: Error) => void) => void): void;
    each(params: OperatorResultListInstanceEachOptions, callback?: (item: OperatorResultInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of OperatorResultInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: OperatorResultPage) => any): Promise<OperatorResultPage>;
    /**
     * Lists OperatorResultInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { OperatorResultListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: OperatorResultInstance[]) => any): Promise<OperatorResultInstance[]>;
    list(params: OperatorResultListInstanceOptions, callback?: (error: Error | null, items: OperatorResultInstance[]) => any): Promise<OperatorResultInstance[]>;
    /**
     * Retrieve a single page of OperatorResultInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { OperatorResultListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: OperatorResultPage) => any): Promise<OperatorResultPage>;
    page(params: OperatorResultListInstancePageOptions, callback?: (error: Error | null, items: OperatorResultPage) => any): Promise<OperatorResultPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function OperatorResultListInstance(version: V2, transcriptSid: string): OperatorResultListInstance;
export declare class OperatorResultPage extends Page<V2, OperatorResultPayload, OperatorResultResource, OperatorResultInstance> {
    /**
     * Initialize the OperatorResultPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: OperatorResultSolution);
    /**
     * Build an instance of OperatorResultInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: OperatorResultResource): OperatorResultInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
