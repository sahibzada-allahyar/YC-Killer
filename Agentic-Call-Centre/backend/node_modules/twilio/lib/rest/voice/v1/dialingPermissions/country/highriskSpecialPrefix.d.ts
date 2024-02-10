/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
/**
 * Options to pass to each
 */
export interface HighriskSpecialPrefixListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: HighriskSpecialPrefixInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface HighriskSpecialPrefixListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface HighriskSpecialPrefixListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface HighriskSpecialPrefixSolution {
    isoCode: string;
}
export interface HighriskSpecialPrefixListInstance {
    _version: V1;
    _solution: HighriskSpecialPrefixSolution;
    _uri: string;
    /**
     * Streams HighriskSpecialPrefixInstance records from the API.
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
     * @param { HighriskSpecialPrefixListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: HighriskSpecialPrefixInstance, done: (err?: Error) => void) => void): void;
    each(params: HighriskSpecialPrefixListInstanceEachOptions, callback?: (item: HighriskSpecialPrefixInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of HighriskSpecialPrefixInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: HighriskSpecialPrefixPage) => any): Promise<HighriskSpecialPrefixPage>;
    /**
     * Lists HighriskSpecialPrefixInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { HighriskSpecialPrefixListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: HighriskSpecialPrefixInstance[]) => any): Promise<HighriskSpecialPrefixInstance[]>;
    list(params: HighriskSpecialPrefixListInstanceOptions, callback?: (error: Error | null, items: HighriskSpecialPrefixInstance[]) => any): Promise<HighriskSpecialPrefixInstance[]>;
    /**
     * Retrieve a single page of HighriskSpecialPrefixInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { HighriskSpecialPrefixListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: HighriskSpecialPrefixPage) => any): Promise<HighriskSpecialPrefixPage>;
    page(params: HighriskSpecialPrefixListInstancePageOptions, callback?: (error: Error | null, items: HighriskSpecialPrefixPage) => any): Promise<HighriskSpecialPrefixPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function HighriskSpecialPrefixListInstance(version: V1, isoCode: string): HighriskSpecialPrefixListInstance;
interface HighriskSpecialPrefixPayload extends TwilioResponsePayload {
    content: HighriskSpecialPrefixResource[];
}
interface HighriskSpecialPrefixResource {
    prefix: string;
}
export declare class HighriskSpecialPrefixInstance {
    protected _version: V1;
    constructor(_version: V1, payload: HighriskSpecialPrefixResource, isoCode: string);
    /**
     * A prefix is a contiguous number range for a block of E.164 numbers that includes the E.164 assigned country code. For example, a North American Numbering Plan prefix like `+1510720` written like `+1(510) 720` matches all numbers inclusive from `+1(510) 720-0000` to `+1(510) 720-9999`.
     */
    prefix: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        prefix: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class HighriskSpecialPrefixPage extends Page<V1, HighriskSpecialPrefixPayload, HighriskSpecialPrefixResource, HighriskSpecialPrefixInstance> {
    /**
     * Initialize the HighriskSpecialPrefixPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: HighriskSpecialPrefixSolution);
    /**
     * Build an instance of HighriskSpecialPrefixInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: HighriskSpecialPrefixResource): HighriskSpecialPrefixInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
