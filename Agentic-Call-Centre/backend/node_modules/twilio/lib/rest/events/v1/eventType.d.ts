/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to each
 */
export interface EventTypeListInstanceEachOptions {
    /** A string parameter filtering the results to return only the Event Types using a given schema. */
    schemaId?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: EventTypeInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface EventTypeListInstanceOptions {
    /** A string parameter filtering the results to return only the Event Types using a given schema. */
    schemaId?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface EventTypeListInstancePageOptions {
    /** A string parameter filtering the results to return only the Event Types using a given schema. */
    schemaId?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface EventTypeContext {
    /**
     * Fetch a EventTypeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EventTypeInstance
     */
    fetch(callback?: (error: Error | null, item?: EventTypeInstance) => any): Promise<EventTypeInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface EventTypeContextSolution {
    type: string;
}
export declare class EventTypeContextImpl implements EventTypeContext {
    protected _version: V1;
    protected _solution: EventTypeContextSolution;
    protected _uri: string;
    constructor(_version: V1, type: string);
    fetch(callback?: (error: Error | null, item?: EventTypeInstance) => any): Promise<EventTypeInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): EventTypeContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface EventTypePayload extends TwilioResponsePayload {
    types: EventTypeResource[];
}
interface EventTypeResource {
    type: string;
    schema_id: string;
    date_created: Date;
    date_updated: Date;
    description: string;
    url: string;
    links: Record<string, string>;
}
export declare class EventTypeInstance {
    protected _version: V1;
    protected _solution: EventTypeContextSolution;
    protected _context?: EventTypeContext;
    constructor(_version: V1, payload: EventTypeResource, type?: string);
    /**
     * A string that uniquely identifies this Event Type.
     */
    type: string;
    /**
     * A string that uniquely identifies the Schema this Event Type adheres to.
     */
    schemaId: string;
    /**
     * The date that this Event Type was created, given in ISO 8601 format.
     */
    dateCreated: Date;
    /**
     * The date that this Event Type was updated, given in ISO 8601 format.
     */
    dateUpdated: Date;
    /**
     * A human readable description for this Event Type.
     */
    description: string;
    /**
     * The URL of this resource.
     */
    url: string;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a EventTypeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EventTypeInstance
     */
    fetch(callback?: (error: Error | null, item?: EventTypeInstance) => any): Promise<EventTypeInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        type: string;
        schemaId: string;
        dateCreated: Date;
        dateUpdated: Date;
        description: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface EventTypeSolution {
}
export interface EventTypeListInstance {
    _version: V1;
    _solution: EventTypeSolution;
    _uri: string;
    (type: string): EventTypeContext;
    get(type: string): EventTypeContext;
    /**
     * Streams EventTypeInstance records from the API.
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
     * @param { EventTypeListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: EventTypeInstance, done: (err?: Error) => void) => void): void;
    each(params: EventTypeListInstanceEachOptions, callback?: (item: EventTypeInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of EventTypeInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: EventTypePage) => any): Promise<EventTypePage>;
    /**
     * Lists EventTypeInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EventTypeListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: EventTypeInstance[]) => any): Promise<EventTypeInstance[]>;
    list(params: EventTypeListInstanceOptions, callback?: (error: Error | null, items: EventTypeInstance[]) => any): Promise<EventTypeInstance[]>;
    /**
     * Retrieve a single page of EventTypeInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EventTypeListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: EventTypePage) => any): Promise<EventTypePage>;
    page(params: EventTypeListInstancePageOptions, callback?: (error: Error | null, items: EventTypePage) => any): Promise<EventTypePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function EventTypeListInstance(version: V1): EventTypeListInstance;
export declare class EventTypePage extends Page<V1, EventTypePayload, EventTypeResource, EventTypeInstance> {
    /**
     * Initialize the EventTypePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: EventTypeSolution);
    /**
     * Build an instance of EventTypeInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: EventTypeResource): EventTypeInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
