/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type TrustProductsEvaluationsStatus = "compliant" | "noncompliant";
/**
 * Options to pass to create a TrustProductsEvaluationsInstance
 */
export interface TrustProductsEvaluationsListInstanceCreateOptions {
    /** The unique string of a policy that is associated to the customer_profile resource. */
    policySid: string;
}
/**
 * Options to pass to each
 */
export interface TrustProductsEvaluationsListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TrustProductsEvaluationsInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TrustProductsEvaluationsListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TrustProductsEvaluationsListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TrustProductsEvaluationsContext {
    /**
     * Fetch a TrustProductsEvaluationsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrustProductsEvaluationsInstance
     */
    fetch(callback?: (error: Error | null, item?: TrustProductsEvaluationsInstance) => any): Promise<TrustProductsEvaluationsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TrustProductsEvaluationsContextSolution {
    trustProductSid: string;
    sid: string;
}
export declare class TrustProductsEvaluationsContextImpl implements TrustProductsEvaluationsContext {
    protected _version: V1;
    protected _solution: TrustProductsEvaluationsContextSolution;
    protected _uri: string;
    constructor(_version: V1, trustProductSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: TrustProductsEvaluationsInstance) => any): Promise<TrustProductsEvaluationsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TrustProductsEvaluationsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TrustProductsEvaluationsPayload extends TwilioResponsePayload {
    results: TrustProductsEvaluationsResource[];
}
interface TrustProductsEvaluationsResource {
    sid: string;
    account_sid: string;
    policy_sid: string;
    trust_product_sid: string;
    status: TrustProductsEvaluationsStatus;
    results: Array<any>;
    date_created: Date;
    url: string;
}
export declare class TrustProductsEvaluationsInstance {
    protected _version: V1;
    protected _solution: TrustProductsEvaluationsContextSolution;
    protected _context?: TrustProductsEvaluationsContext;
    constructor(_version: V1, payload: TrustProductsEvaluationsResource, trustProductSid: string, sid?: string);
    /**
     * The unique string that identifies the Evaluation resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the trust_product resource.
     */
    accountSid: string;
    /**
     * The unique string of a policy that is associated to the trust_product resource.
     */
    policySid: string;
    /**
     * The unique string that we created to identify the trust_product resource.
     */
    trustProductSid: string;
    status: TrustProductsEvaluationsStatus;
    /**
     * The results of the Evaluation which includes the valid and invalid attributes.
     */
    results: Array<any>;
    dateCreated: Date;
    url: string;
    private get _proxy();
    /**
     * Fetch a TrustProductsEvaluationsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrustProductsEvaluationsInstance
     */
    fetch(callback?: (error: Error | null, item?: TrustProductsEvaluationsInstance) => any): Promise<TrustProductsEvaluationsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        policySid: string;
        trustProductSid: string;
        status: TrustProductsEvaluationsStatus;
        results: any[];
        dateCreated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TrustProductsEvaluationsSolution {
    trustProductSid: string;
}
export interface TrustProductsEvaluationsListInstance {
    _version: V1;
    _solution: TrustProductsEvaluationsSolution;
    _uri: string;
    (sid: string): TrustProductsEvaluationsContext;
    get(sid: string): TrustProductsEvaluationsContext;
    /**
     * Create a TrustProductsEvaluationsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrustProductsEvaluationsInstance
     */
    create(params: TrustProductsEvaluationsListInstanceCreateOptions, callback?: (error: Error | null, item?: TrustProductsEvaluationsInstance) => any): Promise<TrustProductsEvaluationsInstance>;
    /**
     * Streams TrustProductsEvaluationsInstance records from the API.
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
     * @param { TrustProductsEvaluationsListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TrustProductsEvaluationsInstance, done: (err?: Error) => void) => void): void;
    each(params: TrustProductsEvaluationsListInstanceEachOptions, callback?: (item: TrustProductsEvaluationsInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TrustProductsEvaluationsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TrustProductsEvaluationsPage) => any): Promise<TrustProductsEvaluationsPage>;
    /**
     * Lists TrustProductsEvaluationsInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TrustProductsEvaluationsListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TrustProductsEvaluationsInstance[]) => any): Promise<TrustProductsEvaluationsInstance[]>;
    list(params: TrustProductsEvaluationsListInstanceOptions, callback?: (error: Error | null, items: TrustProductsEvaluationsInstance[]) => any): Promise<TrustProductsEvaluationsInstance[]>;
    /**
     * Retrieve a single page of TrustProductsEvaluationsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TrustProductsEvaluationsListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TrustProductsEvaluationsPage) => any): Promise<TrustProductsEvaluationsPage>;
    page(params: TrustProductsEvaluationsListInstancePageOptions, callback?: (error: Error | null, items: TrustProductsEvaluationsPage) => any): Promise<TrustProductsEvaluationsPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TrustProductsEvaluationsListInstance(version: V1, trustProductSid: string): TrustProductsEvaluationsListInstance;
export declare class TrustProductsEvaluationsPage extends Page<V1, TrustProductsEvaluationsPayload, TrustProductsEvaluationsResource, TrustProductsEvaluationsInstance> {
    /**
     * Initialize the TrustProductsEvaluationsPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: TrustProductsEvaluationsSolution);
    /**
     * Build an instance of TrustProductsEvaluationsInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TrustProductsEvaluationsResource): TrustProductsEvaluationsInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
