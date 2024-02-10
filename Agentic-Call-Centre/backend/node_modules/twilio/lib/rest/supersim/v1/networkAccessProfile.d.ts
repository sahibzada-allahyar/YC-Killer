/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { NetworkAccessProfileNetworkListInstance } from "./networkAccessProfile/networkAccessProfileNetwork";
/**
 * Options to pass to update a NetworkAccessProfileInstance
 */
export interface NetworkAccessProfileContextUpdateOptions {
    /** The new unique name of the Network Access Profile. */
    uniqueName?: string;
}
/**
 * Options to pass to create a NetworkAccessProfileInstance
 */
export interface NetworkAccessProfileListInstanceCreateOptions {
    /** An application-defined string that uniquely identifies the resource. It can be used in place of the resource\\\'s `sid` in the URL to address the resource. */
    uniqueName?: string;
    /** List of Network SIDs that this Network Access Profile will allow connections to. */
    networks?: Array<string>;
}
/**
 * Options to pass to each
 */
export interface NetworkAccessProfileListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: NetworkAccessProfileInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface NetworkAccessProfileListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface NetworkAccessProfileListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface NetworkAccessProfileContext {
    networks: NetworkAccessProfileNetworkListInstance;
    /**
     * Fetch a NetworkAccessProfileInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkAccessProfileInstance
     */
    fetch(callback?: (error: Error | null, item?: NetworkAccessProfileInstance) => any): Promise<NetworkAccessProfileInstance>;
    /**
     * Update a NetworkAccessProfileInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkAccessProfileInstance
     */
    update(callback?: (error: Error | null, item?: NetworkAccessProfileInstance) => any): Promise<NetworkAccessProfileInstance>;
    /**
     * Update a NetworkAccessProfileInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkAccessProfileInstance
     */
    update(params: NetworkAccessProfileContextUpdateOptions, callback?: (error: Error | null, item?: NetworkAccessProfileInstance) => any): Promise<NetworkAccessProfileInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface NetworkAccessProfileContextSolution {
    sid: string;
}
export declare class NetworkAccessProfileContextImpl implements NetworkAccessProfileContext {
    protected _version: V1;
    protected _solution: NetworkAccessProfileContextSolution;
    protected _uri: string;
    protected _networks?: NetworkAccessProfileNetworkListInstance;
    constructor(_version: V1, sid: string);
    get networks(): NetworkAccessProfileNetworkListInstance;
    fetch(callback?: (error: Error | null, item?: NetworkAccessProfileInstance) => any): Promise<NetworkAccessProfileInstance>;
    update(params?: NetworkAccessProfileContextUpdateOptions | ((error: Error | null, item?: NetworkAccessProfileInstance) => any), callback?: (error: Error | null, item?: NetworkAccessProfileInstance) => any): Promise<NetworkAccessProfileInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): NetworkAccessProfileContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface NetworkAccessProfilePayload extends TwilioResponsePayload {
    network_access_profiles: NetworkAccessProfileResource[];
}
interface NetworkAccessProfileResource {
    sid: string;
    unique_name: string;
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class NetworkAccessProfileInstance {
    protected _version: V1;
    protected _solution: NetworkAccessProfileContextSolution;
    protected _context?: NetworkAccessProfileContext;
    constructor(_version: V1, payload: NetworkAccessProfileResource, sid?: string);
    /**
     * The unique string that identifies the Network Access Profile resource.
     */
    sid: string;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used in place of the resource\'s `sid` in the URL to address the resource.
     */
    uniqueName: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that the Network Access Profile belongs to.
     */
    accountSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Network Access Profile resource.
     */
    url: string;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a NetworkAccessProfileInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkAccessProfileInstance
     */
    fetch(callback?: (error: Error | null, item?: NetworkAccessProfileInstance) => any): Promise<NetworkAccessProfileInstance>;
    /**
     * Update a NetworkAccessProfileInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkAccessProfileInstance
     */
    update(callback?: (error: Error | null, item?: NetworkAccessProfileInstance) => any): Promise<NetworkAccessProfileInstance>;
    /**
     * Update a NetworkAccessProfileInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkAccessProfileInstance
     */
    update(params: NetworkAccessProfileContextUpdateOptions, callback?: (error: Error | null, item?: NetworkAccessProfileInstance) => any): Promise<NetworkAccessProfileInstance>;
    /**
     * Access the networks.
     */
    networks(): NetworkAccessProfileNetworkListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        uniqueName: string;
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface NetworkAccessProfileSolution {
}
export interface NetworkAccessProfileListInstance {
    _version: V1;
    _solution: NetworkAccessProfileSolution;
    _uri: string;
    (sid: string): NetworkAccessProfileContext;
    get(sid: string): NetworkAccessProfileContext;
    /**
     * Create a NetworkAccessProfileInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkAccessProfileInstance
     */
    create(callback?: (error: Error | null, item?: NetworkAccessProfileInstance) => any): Promise<NetworkAccessProfileInstance>;
    /**
     * Create a NetworkAccessProfileInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NetworkAccessProfileInstance
     */
    create(params: NetworkAccessProfileListInstanceCreateOptions, callback?: (error: Error | null, item?: NetworkAccessProfileInstance) => any): Promise<NetworkAccessProfileInstance>;
    /**
     * Streams NetworkAccessProfileInstance records from the API.
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
     * @param { NetworkAccessProfileListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: NetworkAccessProfileInstance, done: (err?: Error) => void) => void): void;
    each(params: NetworkAccessProfileListInstanceEachOptions, callback?: (item: NetworkAccessProfileInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of NetworkAccessProfileInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: NetworkAccessProfilePage) => any): Promise<NetworkAccessProfilePage>;
    /**
     * Lists NetworkAccessProfileInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { NetworkAccessProfileListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: NetworkAccessProfileInstance[]) => any): Promise<NetworkAccessProfileInstance[]>;
    list(params: NetworkAccessProfileListInstanceOptions, callback?: (error: Error | null, items: NetworkAccessProfileInstance[]) => any): Promise<NetworkAccessProfileInstance[]>;
    /**
     * Retrieve a single page of NetworkAccessProfileInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { NetworkAccessProfileListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: NetworkAccessProfilePage) => any): Promise<NetworkAccessProfilePage>;
    page(params: NetworkAccessProfileListInstancePageOptions, callback?: (error: Error | null, items: NetworkAccessProfilePage) => any): Promise<NetworkAccessProfilePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function NetworkAccessProfileListInstance(version: V1): NetworkAccessProfileListInstance;
export declare class NetworkAccessProfilePage extends Page<V1, NetworkAccessProfilePayload, NetworkAccessProfileResource, NetworkAccessProfileInstance> {
    /**
     * Initialize the NetworkAccessProfilePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: NetworkAccessProfileSolution);
    /**
     * Build an instance of NetworkAccessProfileInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: NetworkAccessProfileResource): NetworkAccessProfileInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
