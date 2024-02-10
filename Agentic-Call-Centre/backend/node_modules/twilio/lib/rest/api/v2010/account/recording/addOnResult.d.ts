/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2010 from "../../../V2010";
import { PayloadListInstance } from "./addOnResult/payload";
export type AddOnResultStatus = "canceled" | "completed" | "deleted" | "failed" | "in-progress" | "init" | "processing" | "queued";
/**
 * Options to pass to each
 */
export interface AddOnResultListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AddOnResultInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AddOnResultListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AddOnResultListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AddOnResultContext {
    payloads: PayloadListInstance;
    /**
     * Remove a AddOnResultInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AddOnResultInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AddOnResultInstance
     */
    fetch(callback?: (error: Error | null, item?: AddOnResultInstance) => any): Promise<AddOnResultInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AddOnResultContextSolution {
    accountSid: string;
    referenceSid: string;
    sid: string;
}
export declare class AddOnResultContextImpl implements AddOnResultContext {
    protected _version: V2010;
    protected _solution: AddOnResultContextSolution;
    protected _uri: string;
    protected _payloads?: PayloadListInstance;
    constructor(_version: V2010, accountSid: string, referenceSid: string, sid: string);
    get payloads(): PayloadListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AddOnResultInstance) => any): Promise<AddOnResultInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AddOnResultContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AddOnResultPayload extends TwilioResponsePayload {
    add_on_results: AddOnResultResource[];
}
interface AddOnResultResource {
    sid: string;
    account_sid: string;
    status: AddOnResultStatus;
    add_on_sid: string;
    add_on_configuration_sid: string;
    date_created: Date;
    date_updated: Date;
    date_completed: Date;
    reference_sid: string;
    subresource_uris: Record<string, string>;
}
export declare class AddOnResultInstance {
    protected _version: V2010;
    protected _solution: AddOnResultContextSolution;
    protected _context?: AddOnResultContext;
    constructor(_version: V2010, payload: AddOnResultResource, accountSid: string, referenceSid: string, sid?: string);
    /**
     * The unique string that that we created to identify the Recording AddOnResult resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Recording AddOnResult resource.
     */
    accountSid: string;
    status: AddOnResultStatus;
    /**
     * The SID of the Add-on to which the result belongs.
     */
    addOnSid: string;
    /**
     * The SID of the Add-on configuration.
     */
    addOnConfigurationSid: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The date and time in GMT that the result was completed specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCompleted: Date;
    /**
     * The SID of the recording to which the AddOnResult resource belongs.
     */
    referenceSid: string;
    /**
     * A list of related resources identified by their relative URIs.
     */
    subresourceUris: Record<string, string>;
    private get _proxy();
    /**
     * Remove a AddOnResultInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AddOnResultInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AddOnResultInstance
     */
    fetch(callback?: (error: Error | null, item?: AddOnResultInstance) => any): Promise<AddOnResultInstance>;
    /**
     * Access the payloads.
     */
    payloads(): PayloadListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        status: AddOnResultStatus;
        addOnSid: string;
        addOnConfigurationSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        dateCompleted: Date;
        referenceSid: string;
        subresourceUris: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AddOnResultSolution {
    accountSid: string;
    referenceSid: string;
}
export interface AddOnResultListInstance {
    _version: V2010;
    _solution: AddOnResultSolution;
    _uri: string;
    (sid: string): AddOnResultContext;
    get(sid: string): AddOnResultContext;
    /**
     * Streams AddOnResultInstance records from the API.
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
     * @param { AddOnResultListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AddOnResultInstance, done: (err?: Error) => void) => void): void;
    each(params: AddOnResultListInstanceEachOptions, callback?: (item: AddOnResultInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AddOnResultInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AddOnResultPage) => any): Promise<AddOnResultPage>;
    /**
     * Lists AddOnResultInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AddOnResultListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AddOnResultInstance[]) => any): Promise<AddOnResultInstance[]>;
    list(params: AddOnResultListInstanceOptions, callback?: (error: Error | null, items: AddOnResultInstance[]) => any): Promise<AddOnResultInstance[]>;
    /**
     * Retrieve a single page of AddOnResultInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AddOnResultListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AddOnResultPage) => any): Promise<AddOnResultPage>;
    page(params: AddOnResultListInstancePageOptions, callback?: (error: Error | null, items: AddOnResultPage) => any): Promise<AddOnResultPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AddOnResultListInstance(version: V2010, accountSid: string, referenceSid: string): AddOnResultListInstance;
export declare class AddOnResultPage extends Page<V2010, AddOnResultPayload, AddOnResultResource, AddOnResultInstance> {
    /**
     * Initialize the AddOnResultPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: AddOnResultSolution);
    /**
     * Build an instance of AddOnResultInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AddOnResultResource): AddOnResultInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
