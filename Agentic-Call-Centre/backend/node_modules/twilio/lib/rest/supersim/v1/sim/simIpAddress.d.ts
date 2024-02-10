/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type SimIpAddressIpAddressVersion = "IPv4" | "IPv6";
/**
 * Options to pass to each
 */
export interface SimIpAddressListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SimIpAddressInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SimIpAddressListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SimIpAddressListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SimIpAddressSolution {
    simSid: string;
}
export interface SimIpAddressListInstance {
    _version: V1;
    _solution: SimIpAddressSolution;
    _uri: string;
    /**
     * Streams SimIpAddressInstance records from the API.
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
     * @param { SimIpAddressListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SimIpAddressInstance, done: (err?: Error) => void) => void): void;
    each(params: SimIpAddressListInstanceEachOptions, callback?: (item: SimIpAddressInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SimIpAddressInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SimIpAddressPage) => any): Promise<SimIpAddressPage>;
    /**
     * Lists SimIpAddressInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SimIpAddressListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SimIpAddressInstance[]) => any): Promise<SimIpAddressInstance[]>;
    list(params: SimIpAddressListInstanceOptions, callback?: (error: Error | null, items: SimIpAddressInstance[]) => any): Promise<SimIpAddressInstance[]>;
    /**
     * Retrieve a single page of SimIpAddressInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SimIpAddressListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SimIpAddressPage) => any): Promise<SimIpAddressPage>;
    page(params: SimIpAddressListInstancePageOptions, callback?: (error: Error | null, items: SimIpAddressPage) => any): Promise<SimIpAddressPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SimIpAddressListInstance(version: V1, simSid: string): SimIpAddressListInstance;
interface SimIpAddressPayload extends TwilioResponsePayload {
    ip_addresses: SimIpAddressResource[];
}
interface SimIpAddressResource {
    ip_address: string;
    ip_address_version: SimIpAddressIpAddressVersion;
}
export declare class SimIpAddressInstance {
    protected _version: V1;
    constructor(_version: V1, payload: SimIpAddressResource, simSid: string);
    /**
     * IP address assigned to the given Super SIM
     */
    ipAddress: string;
    ipAddressVersion: SimIpAddressIpAddressVersion;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        ipAddress: string;
        ipAddressVersion: SimIpAddressIpAddressVersion;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class SimIpAddressPage extends Page<V1, SimIpAddressPayload, SimIpAddressResource, SimIpAddressInstance> {
    /**
     * Initialize the SimIpAddressPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SimIpAddressSolution);
    /**
     * Build an instance of SimIpAddressInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SimIpAddressResource): SimIpAddressInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
