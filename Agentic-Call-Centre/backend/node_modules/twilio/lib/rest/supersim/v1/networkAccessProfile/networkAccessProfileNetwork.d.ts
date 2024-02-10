/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to create a NetworkAccessProfileNetworkInstance
 */
export interface NetworkAccessProfileNetworkListInstanceCreateOptions {
    /** The SID of the Network resource to be added to the Network Access Profile resource. */
    network: string;
}
/**
 * Options to pass to each
 */
export interface NetworkAccessProfileNetworkListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: NetworkAccessProfileNetworkInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface NetworkAccessProfileNetworkListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface NetworkAccessProfileNetworkListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface NetworkAccessProfileNetworkContext {
    /**
     * Remove a NetworkAccessProfileNetworkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a NetworkAccessProfileNetworkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkAccessProfileNetworkInstance
     */
    fetch(callback?: (error: Error | null, item?: NetworkAccessProfileNetworkInstance) => any): Promise<NetworkAccessProfileNetworkInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface NetworkAccessProfileNetworkContextSolution {
    networkAccessProfileSid: string;
    sid: string;
}
export declare class NetworkAccessProfileNetworkContextImpl implements NetworkAccessProfileNetworkContext {
    protected _version: V1;
    protected _solution: NetworkAccessProfileNetworkContextSolution;
    protected _uri: string;
    constructor(_version: V1, networkAccessProfileSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: NetworkAccessProfileNetworkInstance) => any): Promise<NetworkAccessProfileNetworkInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): NetworkAccessProfileNetworkContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface NetworkAccessProfileNetworkPayload extends TwilioResponsePayload {
    networks: NetworkAccessProfileNetworkResource[];
}
interface NetworkAccessProfileNetworkResource {
    sid: string;
    network_access_profile_sid: string;
    friendly_name: string;
    iso_country: string;
    identifiers: Array<any>;
    url: string;
}
export declare class NetworkAccessProfileNetworkInstance {
    protected _version: V1;
    protected _solution: NetworkAccessProfileNetworkContextSolution;
    protected _context?: NetworkAccessProfileNetworkContext;
    constructor(_version: V1, payload: NetworkAccessProfileNetworkResource, networkAccessProfileSid: string, sid?: string);
    /**
     * The unique string that identifies the Network resource.
     */
    sid: string;
    /**
     * The unique string that identifies the Network resource\'s Network Access Profile resource.
     */
    networkAccessProfileSid: string;
    /**
     * A human readable identifier of the Network this resource refers to.
     */
    friendlyName: string;
    /**
     * The [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the Network resource.
     */
    isoCountry: string;
    /**
     * Array of objects identifying the [MCC-MNCs](https://en.wikipedia.org/wiki/Mobile_country_code) that are included in the Network resource.
     */
    identifiers: Array<any>;
    /**
     * The absolute URL of the Network resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a NetworkAccessProfileNetworkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a NetworkAccessProfileNetworkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkAccessProfileNetworkInstance
     */
    fetch(callback?: (error: Error | null, item?: NetworkAccessProfileNetworkInstance) => any): Promise<NetworkAccessProfileNetworkInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        networkAccessProfileSid: string;
        friendlyName: string;
        isoCountry: string;
        identifiers: any[];
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface NetworkAccessProfileNetworkSolution {
    networkAccessProfileSid: string;
}
export interface NetworkAccessProfileNetworkListInstance {
    _version: V1;
    _solution: NetworkAccessProfileNetworkSolution;
    _uri: string;
    (sid: string): NetworkAccessProfileNetworkContext;
    get(sid: string): NetworkAccessProfileNetworkContext;
    /**
     * Create a NetworkAccessProfileNetworkInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkAccessProfileNetworkInstance
     */
    create(params: NetworkAccessProfileNetworkListInstanceCreateOptions, callback?: (error: Error | null, item?: NetworkAccessProfileNetworkInstance) => any): Promise<NetworkAccessProfileNetworkInstance>;
    /**
     * Streams NetworkAccessProfileNetworkInstance records from the API.
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
     * @param { NetworkAccessProfileNetworkListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: NetworkAccessProfileNetworkInstance, done: (err?: Error) => void) => void): void;
    each(params: NetworkAccessProfileNetworkListInstanceEachOptions, callback?: (item: NetworkAccessProfileNetworkInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of NetworkAccessProfileNetworkInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: NetworkAccessProfileNetworkPage) => any): Promise<NetworkAccessProfileNetworkPage>;
    /**
     * Lists NetworkAccessProfileNetworkInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { NetworkAccessProfileNetworkListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: NetworkAccessProfileNetworkInstance[]) => any): Promise<NetworkAccessProfileNetworkInstance[]>;
    list(params: NetworkAccessProfileNetworkListInstanceOptions, callback?: (error: Error | null, items: NetworkAccessProfileNetworkInstance[]) => any): Promise<NetworkAccessProfileNetworkInstance[]>;
    /**
     * Retrieve a single page of NetworkAccessProfileNetworkInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { NetworkAccessProfileNetworkListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: NetworkAccessProfileNetworkPage) => any): Promise<NetworkAccessProfileNetworkPage>;
    page(params: NetworkAccessProfileNetworkListInstancePageOptions, callback?: (error: Error | null, items: NetworkAccessProfileNetworkPage) => any): Promise<NetworkAccessProfileNetworkPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function NetworkAccessProfileNetworkListInstance(version: V1, networkAccessProfileSid: string): NetworkAccessProfileNetworkListInstance;
export declare class NetworkAccessProfileNetworkPage extends Page<V1, NetworkAccessProfileNetworkPayload, NetworkAccessProfileNetworkResource, NetworkAccessProfileNetworkInstance> {
    /**
     * Initialize the NetworkAccessProfileNetworkPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: NetworkAccessProfileNetworkSolution);
    /**
     * Build an instance of NetworkAccessProfileNetworkInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: NetworkAccessProfileNetworkResource): NetworkAccessProfileNetworkInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
