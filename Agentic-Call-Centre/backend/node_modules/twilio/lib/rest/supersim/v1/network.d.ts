/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to each
 */
export interface NetworkListInstanceEachOptions {
    /** The [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the Network resources to read. */
    isoCountry?: string;
    /** The \'mobile country code\' of a country. Network resources with this `mcc` in their `identifiers` will be read. */
    mcc?: string;
    /** The \'mobile network code\' of a mobile operator network. Network resources with this `mnc` in their `identifiers` will be read. */
    mnc?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: NetworkInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface NetworkListInstanceOptions {
    /** The [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the Network resources to read. */
    isoCountry?: string;
    /** The \'mobile country code\' of a country. Network resources with this `mcc` in their `identifiers` will be read. */
    mcc?: string;
    /** The \'mobile network code\' of a mobile operator network. Network resources with this `mnc` in their `identifiers` will be read. */
    mnc?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface NetworkListInstancePageOptions {
    /** The [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the Network resources to read. */
    isoCountry?: string;
    /** The \'mobile country code\' of a country. Network resources with this `mcc` in their `identifiers` will be read. */
    mcc?: string;
    /** The \'mobile network code\' of a mobile operator network. Network resources with this `mnc` in their `identifiers` will be read. */
    mnc?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface NetworkContext {
    /**
     * Fetch a NetworkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkInstance
     */
    fetch(callback?: (error: Error | null, item?: NetworkInstance) => any): Promise<NetworkInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface NetworkContextSolution {
    sid: string;
}
export declare class NetworkContextImpl implements NetworkContext {
    protected _version: V1;
    protected _solution: NetworkContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: NetworkInstance) => any): Promise<NetworkInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): NetworkContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface NetworkPayload extends TwilioResponsePayload {
    networks: NetworkResource[];
}
interface NetworkResource {
    sid: string;
    friendly_name: string;
    url: string;
    iso_country: string;
    identifiers: Array<any>;
}
export declare class NetworkInstance {
    protected _version: V1;
    protected _solution: NetworkContextSolution;
    protected _context?: NetworkContext;
    constructor(_version: V1, payload: NetworkResource, sid?: string);
    /**
     * The unique string that we created to identify the Network resource.
     */
    sid: string;
    /**
     * A human readable identifier of this resource.
     */
    friendlyName: string;
    /**
     * The absolute URL of the Network resource.
     */
    url: string;
    /**
     * The [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the Network resource.
     */
    isoCountry: string;
    /**
     * Array of objects identifying the [MCC-MNCs](https://en.wikipedia.org/wiki/Mobile_country_code) that are included in the Network resource.
     */
    identifiers: Array<any>;
    private get _proxy();
    /**
     * Fetch a NetworkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkInstance
     */
    fetch(callback?: (error: Error | null, item?: NetworkInstance) => any): Promise<NetworkInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        friendlyName: string;
        url: string;
        isoCountry: string;
        identifiers: any[];
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface NetworkSolution {
}
export interface NetworkListInstance {
    _version: V1;
    _solution: NetworkSolution;
    _uri: string;
    (sid: string): NetworkContext;
    get(sid: string): NetworkContext;
    /**
     * Streams NetworkInstance records from the API.
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
     * @param { NetworkListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: NetworkInstance, done: (err?: Error) => void) => void): void;
    each(params: NetworkListInstanceEachOptions, callback?: (item: NetworkInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of NetworkInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: NetworkPage) => any): Promise<NetworkPage>;
    /**
     * Lists NetworkInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { NetworkListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: NetworkInstance[]) => any): Promise<NetworkInstance[]>;
    list(params: NetworkListInstanceOptions, callback?: (error: Error | null, items: NetworkInstance[]) => any): Promise<NetworkInstance[]>;
    /**
     * Retrieve a single page of NetworkInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { NetworkListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: NetworkPage) => any): Promise<NetworkPage>;
    page(params: NetworkListInstancePageOptions, callback?: (error: Error | null, items: NetworkPage) => any): Promise<NetworkPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function NetworkListInstance(version: V1): NetworkListInstance;
export declare class NetworkPage extends Page<V1, NetworkPayload, NetworkResource, NetworkInstance> {
    /**
     * Initialize the NetworkPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: NetworkSolution);
    /**
     * Build an instance of NetworkInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: NetworkResource): NetworkInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
