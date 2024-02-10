/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2 from "../../../V2";
export type EvaluationStatus = "compliant" | "noncompliant";
/**
 * Options to pass to each
 */
export interface EvaluationListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: EvaluationInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface EvaluationListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface EvaluationListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface EvaluationContext {
    /**
     * Fetch a EvaluationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EvaluationInstance
     */
    fetch(callback?: (error: Error | null, item?: EvaluationInstance) => any): Promise<EvaluationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface EvaluationContextSolution {
    bundleSid: string;
    sid: string;
}
export declare class EvaluationContextImpl implements EvaluationContext {
    protected _version: V2;
    protected _solution: EvaluationContextSolution;
    protected _uri: string;
    constructor(_version: V2, bundleSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: EvaluationInstance) => any): Promise<EvaluationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): EvaluationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface EvaluationPayload extends TwilioResponsePayload {
    results: EvaluationResource[];
}
interface EvaluationResource {
    sid: string;
    account_sid: string;
    regulation_sid: string;
    bundle_sid: string;
    status: EvaluationStatus;
    results: Array<any>;
    date_created: Date;
    url: string;
}
export declare class EvaluationInstance {
    protected _version: V2;
    protected _solution: EvaluationContextSolution;
    protected _context?: EvaluationContext;
    constructor(_version: V2, payload: EvaluationResource, bundleSid: string, sid?: string);
    /**
     * The unique string that identifies the Evaluation resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Bundle resource.
     */
    accountSid: string;
    /**
     * The unique string of a regulation that is associated to the Bundle resource.
     */
    regulationSid: string;
    /**
     * The unique string that we created to identify the Bundle resource.
     */
    bundleSid: string;
    status: EvaluationStatus;
    /**
     * The results of the Evaluation which includes the valid and invalid attributes.
     */
    results: Array<any>;
    dateCreated: Date;
    url: string;
    private get _proxy();
    /**
     * Fetch a EvaluationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EvaluationInstance
     */
    fetch(callback?: (error: Error | null, item?: EvaluationInstance) => any): Promise<EvaluationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        regulationSid: string;
        bundleSid: string;
        status: EvaluationStatus;
        results: any[];
        dateCreated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface EvaluationSolution {
    bundleSid: string;
}
export interface EvaluationListInstance {
    _version: V2;
    _solution: EvaluationSolution;
    _uri: string;
    (sid: string): EvaluationContext;
    get(sid: string): EvaluationContext;
    /**
     * Create a EvaluationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EvaluationInstance
     */
    create(callback?: (error: Error | null, item?: EvaluationInstance) => any): Promise<EvaluationInstance>;
    /**
     * Streams EvaluationInstance records from the API.
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
     * @param { EvaluationListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: EvaluationInstance, done: (err?: Error) => void) => void): void;
    each(params: EvaluationListInstanceEachOptions, callback?: (item: EvaluationInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of EvaluationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: EvaluationPage) => any): Promise<EvaluationPage>;
    /**
     * Lists EvaluationInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EvaluationListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: EvaluationInstance[]) => any): Promise<EvaluationInstance[]>;
    list(params: EvaluationListInstanceOptions, callback?: (error: Error | null, items: EvaluationInstance[]) => any): Promise<EvaluationInstance[]>;
    /**
     * Retrieve a single page of EvaluationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EvaluationListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: EvaluationPage) => any): Promise<EvaluationPage>;
    page(params: EvaluationListInstancePageOptions, callback?: (error: Error | null, items: EvaluationPage) => any): Promise<EvaluationPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function EvaluationListInstance(version: V2, bundleSid: string): EvaluationListInstance;
export declare class EvaluationPage extends Page<V2, EvaluationPayload, EvaluationResource, EvaluationInstance> {
    /**
     * Initialize the EvaluationPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: EvaluationSolution);
    /**
     * Build an instance of EvaluationInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: EvaluationResource): EvaluationInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
