/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
export type BindingBindingType = "gcm" | "apn" | "fcm";
/**
 * Options to pass to each
 */
export interface BindingListInstanceEachOptions {
    /**  */
    bindingType?: Array<BindingBindingType>;
    /**  */
    identity?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: BindingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface BindingListInstanceOptions {
    /**  */
    bindingType?: Array<BindingBindingType>;
    /**  */
    identity?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface BindingListInstancePageOptions {
    /**  */
    bindingType?: Array<BindingBindingType>;
    /**  */
    identity?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface BindingContext {
    /**
     * Remove a BindingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a BindingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BindingInstance
     */
    fetch(callback?: (error: Error | null, item?: BindingInstance) => any): Promise<BindingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface BindingContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class BindingContextImpl implements BindingContext {
    protected _version: V2;
    protected _solution: BindingContextSolution;
    protected _uri: string;
    constructor(_version: V2, serviceSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: BindingInstance) => any): Promise<BindingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): BindingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface BindingPayload extends TwilioResponsePayload {
    bindings: BindingResource[];
}
interface BindingResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    date_created: Date;
    date_updated: Date;
    endpoint: string;
    identity: string;
    credential_sid: string;
    binding_type: BindingBindingType;
    message_types: Array<string>;
    url: string;
    links: Record<string, string>;
}
export declare class BindingInstance {
    protected _version: V2;
    protected _solution: BindingContextSolution;
    protected _context?: BindingContext;
    constructor(_version: V2, payload: BindingResource, serviceSid: string, sid?: string);
    sid: string;
    accountSid: string;
    serviceSid: string;
    dateCreated: Date;
    dateUpdated: Date;
    endpoint: string;
    identity: string;
    credentialSid: string;
    bindingType: BindingBindingType;
    messageTypes: Array<string>;
    url: string;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a BindingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a BindingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BindingInstance
     */
    fetch(callback?: (error: Error | null, item?: BindingInstance) => any): Promise<BindingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        endpoint: string;
        identity: string;
        credentialSid: string;
        bindingType: BindingBindingType;
        messageTypes: string[];
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface BindingSolution {
    serviceSid: string;
}
export interface BindingListInstance {
    _version: V2;
    _solution: BindingSolution;
    _uri: string;
    (sid: string): BindingContext;
    get(sid: string): BindingContext;
    /**
     * Streams BindingInstance records from the API.
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
     * @param { BindingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: BindingInstance, done: (err?: Error) => void) => void): void;
    each(params: BindingListInstanceEachOptions, callback?: (item: BindingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of BindingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: BindingPage) => any): Promise<BindingPage>;
    /**
     * Lists BindingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BindingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: BindingInstance[]) => any): Promise<BindingInstance[]>;
    list(params: BindingListInstanceOptions, callback?: (error: Error | null, items: BindingInstance[]) => any): Promise<BindingInstance[]>;
    /**
     * Retrieve a single page of BindingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BindingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: BindingPage) => any): Promise<BindingPage>;
    page(params: BindingListInstancePageOptions, callback?: (error: Error | null, items: BindingPage) => any): Promise<BindingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BindingListInstance(version: V2, serviceSid: string): BindingListInstance;
export declare class BindingPage extends Page<V2, BindingPayload, BindingResource, BindingInstance> {
    /**
     * Initialize the BindingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: BindingSolution);
    /**
     * Build an instance of BindingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: BindingResource): BindingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
