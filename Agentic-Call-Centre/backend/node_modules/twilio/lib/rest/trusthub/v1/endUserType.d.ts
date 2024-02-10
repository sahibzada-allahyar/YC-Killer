/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to each
 */
export interface EndUserTypeListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: EndUserTypeInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface EndUserTypeListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface EndUserTypeListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface EndUserTypeContext {
    /**
     * Fetch a EndUserTypeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EndUserTypeInstance
     */
    fetch(callback?: (error: Error | null, item?: EndUserTypeInstance) => any): Promise<EndUserTypeInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface EndUserTypeContextSolution {
    sid: string;
}
export declare class EndUserTypeContextImpl implements EndUserTypeContext {
    protected _version: V1;
    protected _solution: EndUserTypeContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: EndUserTypeInstance) => any): Promise<EndUserTypeInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): EndUserTypeContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface EndUserTypePayload extends TwilioResponsePayload {
    end_user_types: EndUserTypeResource[];
}
interface EndUserTypeResource {
    sid: string;
    friendly_name: string;
    machine_name: string;
    fields: Array<any>;
    url: string;
}
export declare class EndUserTypeInstance {
    protected _version: V1;
    protected _solution: EndUserTypeContextSolution;
    protected _context?: EndUserTypeContext;
    constructor(_version: V1, payload: EndUserTypeResource, sid?: string);
    /**
     * The unique string that identifies the End-User Type resource.
     */
    sid: string;
    /**
     * A human-readable description that is assigned to describe the End-User Type resource. Examples can include first name, last name, email, business name, etc
     */
    friendlyName: string;
    /**
     * A machine-readable description of the End-User Type resource. Examples can include first_name, last_name, email, business_name, etc.
     */
    machineName: string;
    /**
     * The required information for creating an End-User. The required fields will change as regulatory needs change and will differ for businesses and individuals.
     */
    fields: Array<any>;
    /**
     * The absolute URL of the End-User Type resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a EndUserTypeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EndUserTypeInstance
     */
    fetch(callback?: (error: Error | null, item?: EndUserTypeInstance) => any): Promise<EndUserTypeInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        friendlyName: string;
        machineName: string;
        fields: any[];
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface EndUserTypeSolution {
}
export interface EndUserTypeListInstance {
    _version: V1;
    _solution: EndUserTypeSolution;
    _uri: string;
    (sid: string): EndUserTypeContext;
    get(sid: string): EndUserTypeContext;
    /**
     * Streams EndUserTypeInstance records from the API.
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
     * @param { EndUserTypeListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: EndUserTypeInstance, done: (err?: Error) => void) => void): void;
    each(params: EndUserTypeListInstanceEachOptions, callback?: (item: EndUserTypeInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of EndUserTypeInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: EndUserTypePage) => any): Promise<EndUserTypePage>;
    /**
     * Lists EndUserTypeInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EndUserTypeListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: EndUserTypeInstance[]) => any): Promise<EndUserTypeInstance[]>;
    list(params: EndUserTypeListInstanceOptions, callback?: (error: Error | null, items: EndUserTypeInstance[]) => any): Promise<EndUserTypeInstance[]>;
    /**
     * Retrieve a single page of EndUserTypeInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EndUserTypeListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: EndUserTypePage) => any): Promise<EndUserTypePage>;
    page(params: EndUserTypeListInstancePageOptions, callback?: (error: Error | null, items: EndUserTypePage) => any): Promise<EndUserTypePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function EndUserTypeListInstance(version: V1): EndUserTypeListInstance;
export declare class EndUserTypePage extends Page<V1, EndUserTypePayload, EndUserTypeResource, EndUserTypeInstance> {
    /**
     * Initialize the EndUserTypePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: EndUserTypeSolution);
    /**
     * Build an instance of EndUserTypeInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: EndUserTypeResource): EndUserTypeInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
