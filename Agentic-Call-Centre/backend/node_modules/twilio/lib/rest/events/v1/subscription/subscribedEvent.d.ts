/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to update a SubscribedEventInstance
 */
export interface SubscribedEventContextUpdateOptions {
    /** The schema version that the Subscription should use. */
    schemaVersion?: number;
}
/**
 * Options to pass to create a SubscribedEventInstance
 */
export interface SubscribedEventListInstanceCreateOptions {
    /** Type of event being subscribed to. */
    type: string;
    /** The schema version that the Subscription should use. */
    schemaVersion?: number;
}
/**
 * Options to pass to each
 */
export interface SubscribedEventListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SubscribedEventInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SubscribedEventListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SubscribedEventListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SubscribedEventContext {
    /**
     * Remove a SubscribedEventInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SubscribedEventInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscribedEventInstance
     */
    fetch(callback?: (error: Error | null, item?: SubscribedEventInstance) => any): Promise<SubscribedEventInstance>;
    /**
     * Update a SubscribedEventInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscribedEventInstance
     */
    update(callback?: (error: Error | null, item?: SubscribedEventInstance) => any): Promise<SubscribedEventInstance>;
    /**
     * Update a SubscribedEventInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscribedEventInstance
     */
    update(params: SubscribedEventContextUpdateOptions, callback?: (error: Error | null, item?: SubscribedEventInstance) => any): Promise<SubscribedEventInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SubscribedEventContextSolution {
    subscriptionSid: string;
    type: string;
}
export declare class SubscribedEventContextImpl implements SubscribedEventContext {
    protected _version: V1;
    protected _solution: SubscribedEventContextSolution;
    protected _uri: string;
    constructor(_version: V1, subscriptionSid: string, type: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SubscribedEventInstance) => any): Promise<SubscribedEventInstance>;
    update(params?: SubscribedEventContextUpdateOptions | ((error: Error | null, item?: SubscribedEventInstance) => any), callback?: (error: Error | null, item?: SubscribedEventInstance) => any): Promise<SubscribedEventInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SubscribedEventContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SubscribedEventPayload extends TwilioResponsePayload {
    types: SubscribedEventResource[];
}
interface SubscribedEventResource {
    account_sid: string;
    type: string;
    schema_version: number;
    subscription_sid: string;
    url: string;
}
export declare class SubscribedEventInstance {
    protected _version: V1;
    protected _solution: SubscribedEventContextSolution;
    protected _context?: SubscribedEventContext;
    constructor(_version: V1, payload: SubscribedEventResource, subscriptionSid: string, type?: string);
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * Type of event being subscribed to.
     */
    type: string;
    /**
     * The schema version that the Subscription should use.
     */
    schemaVersion: number;
    /**
     * The unique SID identifier of the Subscription.
     */
    subscriptionSid: string;
    /**
     * The URL of this resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a SubscribedEventInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SubscribedEventInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscribedEventInstance
     */
    fetch(callback?: (error: Error | null, item?: SubscribedEventInstance) => any): Promise<SubscribedEventInstance>;
    /**
     * Update a SubscribedEventInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscribedEventInstance
     */
    update(callback?: (error: Error | null, item?: SubscribedEventInstance) => any): Promise<SubscribedEventInstance>;
    /**
     * Update a SubscribedEventInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscribedEventInstance
     */
    update(params: SubscribedEventContextUpdateOptions, callback?: (error: Error | null, item?: SubscribedEventInstance) => any): Promise<SubscribedEventInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        type: string;
        schemaVersion: number;
        subscriptionSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SubscribedEventSolution {
    subscriptionSid: string;
}
export interface SubscribedEventListInstance {
    _version: V1;
    _solution: SubscribedEventSolution;
    _uri: string;
    (type: string): SubscribedEventContext;
    get(type: string): SubscribedEventContext;
    /**
     * Create a SubscribedEventInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscribedEventInstance
     */
    create(params: SubscribedEventListInstanceCreateOptions, callback?: (error: Error | null, item?: SubscribedEventInstance) => any): Promise<SubscribedEventInstance>;
    /**
     * Streams SubscribedEventInstance records from the API.
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
     * @param { SubscribedEventListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SubscribedEventInstance, done: (err?: Error) => void) => void): void;
    each(params: SubscribedEventListInstanceEachOptions, callback?: (item: SubscribedEventInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SubscribedEventInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SubscribedEventPage) => any): Promise<SubscribedEventPage>;
    /**
     * Lists SubscribedEventInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SubscribedEventListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SubscribedEventInstance[]) => any): Promise<SubscribedEventInstance[]>;
    list(params: SubscribedEventListInstanceOptions, callback?: (error: Error | null, items: SubscribedEventInstance[]) => any): Promise<SubscribedEventInstance[]>;
    /**
     * Retrieve a single page of SubscribedEventInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SubscribedEventListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SubscribedEventPage) => any): Promise<SubscribedEventPage>;
    page(params: SubscribedEventListInstancePageOptions, callback?: (error: Error | null, items: SubscribedEventPage) => any): Promise<SubscribedEventPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SubscribedEventListInstance(version: V1, subscriptionSid: string): SubscribedEventListInstance;
export declare class SubscribedEventPage extends Page<V1, SubscribedEventPayload, SubscribedEventResource, SubscribedEventInstance> {
    /**
     * Initialize the SubscribedEventPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SubscribedEventSolution);
    /**
     * Build an instance of SubscribedEventInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SubscribedEventResource): SubscribedEventInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
