/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { SubscribedEventListInstance } from "./subscription/subscribedEvent";
/**
 * Options to pass to update a SubscriptionInstance
 */
export interface SubscriptionContextUpdateOptions {
    /** A human readable description for the Subscription. */
    description?: string;
    /** The SID of the sink that events selected by this subscription should be sent to. Sink must be active for the subscription to be created. */
    sinkSid?: string;
}
/**
 * Options to pass to create a SubscriptionInstance
 */
export interface SubscriptionListInstanceCreateOptions {
    /** A human readable description for the Subscription **This value should not contain PII.** */
    description: string;
    /** The SID of the sink that events selected by this subscription should be sent to. Sink must be active for the subscription to be created. */
    sinkSid: string;
    /** An array of objects containing the subscribed Event Types */
    types: Array<any>;
}
/**
 * Options to pass to each
 */
export interface SubscriptionListInstanceEachOptions {
    /** The SID of the sink that the list of Subscriptions should be filtered by. */
    sinkSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SubscriptionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SubscriptionListInstanceOptions {
    /** The SID of the sink that the list of Subscriptions should be filtered by. */
    sinkSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SubscriptionListInstancePageOptions {
    /** The SID of the sink that the list of Subscriptions should be filtered by. */
    sinkSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SubscriptionContext {
    subscribedEvents: SubscribedEventListInstance;
    /**
     * Remove a SubscriptionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SubscriptionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscriptionInstance
     */
    fetch(callback?: (error: Error | null, item?: SubscriptionInstance) => any): Promise<SubscriptionInstance>;
    /**
     * Update a SubscriptionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscriptionInstance
     */
    update(callback?: (error: Error | null, item?: SubscriptionInstance) => any): Promise<SubscriptionInstance>;
    /**
     * Update a SubscriptionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscriptionInstance
     */
    update(params: SubscriptionContextUpdateOptions, callback?: (error: Error | null, item?: SubscriptionInstance) => any): Promise<SubscriptionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SubscriptionContextSolution {
    sid: string;
}
export declare class SubscriptionContextImpl implements SubscriptionContext {
    protected _version: V1;
    protected _solution: SubscriptionContextSolution;
    protected _uri: string;
    protected _subscribedEvents?: SubscribedEventListInstance;
    constructor(_version: V1, sid: string);
    get subscribedEvents(): SubscribedEventListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SubscriptionInstance) => any): Promise<SubscriptionInstance>;
    update(params?: SubscriptionContextUpdateOptions | ((error: Error | null, item?: SubscriptionInstance) => any), callback?: (error: Error | null, item?: SubscriptionInstance) => any): Promise<SubscriptionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SubscriptionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SubscriptionPayload extends TwilioResponsePayload {
    subscriptions: SubscriptionResource[];
}
interface SubscriptionResource {
    account_sid: string;
    sid: string;
    date_created: Date;
    date_updated: Date;
    description: string;
    sink_sid: string;
    url: string;
    links: Record<string, string>;
}
export declare class SubscriptionInstance {
    protected _version: V1;
    protected _solution: SubscriptionContextSolution;
    protected _context?: SubscriptionContext;
    constructor(_version: V1, payload: SubscriptionResource, sid?: string);
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * A 34 character string that uniquely identifies this Subscription.
     */
    sid: string;
    /**
     * The date that this Subscription was created, given in ISO 8601 format.
     */
    dateCreated: Date;
    /**
     * The date that this Subscription was updated, given in ISO 8601 format.
     */
    dateUpdated: Date;
    /**
     * A human readable description for the Subscription
     */
    description: string;
    /**
     * The SID of the sink that events selected by this subscription should be sent to. Sink must be active for the subscription to be created.
     */
    sinkSid: string;
    /**
     * The URL of this resource.
     */
    url: string;
    /**
     * Contains a dictionary of URL links to nested resources of this Subscription.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a SubscriptionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SubscriptionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscriptionInstance
     */
    fetch(callback?: (error: Error | null, item?: SubscriptionInstance) => any): Promise<SubscriptionInstance>;
    /**
     * Update a SubscriptionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscriptionInstance
     */
    update(callback?: (error: Error | null, item?: SubscriptionInstance) => any): Promise<SubscriptionInstance>;
    /**
     * Update a SubscriptionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscriptionInstance
     */
    update(params: SubscriptionContextUpdateOptions, callback?: (error: Error | null, item?: SubscriptionInstance) => any): Promise<SubscriptionInstance>;
    /**
     * Access the subscribedEvents.
     */
    subscribedEvents(): SubscribedEventListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        sid: string;
        dateCreated: Date;
        dateUpdated: Date;
        description: string;
        sinkSid: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SubscriptionSolution {
}
export interface SubscriptionListInstance {
    _version: V1;
    _solution: SubscriptionSolution;
    _uri: string;
    (sid: string): SubscriptionContext;
    get(sid: string): SubscriptionContext;
    /**
     * Create a SubscriptionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscriptionInstance
     */
    create(params: SubscriptionListInstanceCreateOptions, callback?: (error: Error | null, item?: SubscriptionInstance) => any): Promise<SubscriptionInstance>;
    /**
     * Streams SubscriptionInstance records from the API.
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
     * @param { SubscriptionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SubscriptionInstance, done: (err?: Error) => void) => void): void;
    each(params: SubscriptionListInstanceEachOptions, callback?: (item: SubscriptionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SubscriptionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SubscriptionPage) => any): Promise<SubscriptionPage>;
    /**
     * Lists SubscriptionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SubscriptionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SubscriptionInstance[]) => any): Promise<SubscriptionInstance[]>;
    list(params: SubscriptionListInstanceOptions, callback?: (error: Error | null, items: SubscriptionInstance[]) => any): Promise<SubscriptionInstance[]>;
    /**
     * Retrieve a single page of SubscriptionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SubscriptionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SubscriptionPage) => any): Promise<SubscriptionPage>;
    page(params: SubscriptionListInstancePageOptions, callback?: (error: Error | null, items: SubscriptionPage) => any): Promise<SubscriptionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SubscriptionListInstance(version: V1): SubscriptionListInstance;
export declare class SubscriptionPage extends Page<V1, SubscriptionPayload, SubscriptionResource, SubscriptionInstance> {
    /**
     * Initialize the SubscriptionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SubscriptionSolution);
    /**
     * Build an instance of SubscriptionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SubscriptionResource): SubscriptionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
