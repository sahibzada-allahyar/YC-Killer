/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to update a SourceIpMappingInstance
 */
export interface SourceIpMappingContextUpdateOptions {
    /** The SID of the SIP Domain that the IP Record should be mapped to. */
    sipDomainSid: string;
}
/**
 * Options to pass to create a SourceIpMappingInstance
 */
export interface SourceIpMappingListInstanceCreateOptions {
    /** The Twilio-provided string that uniquely identifies the IP Record resource to map from. */
    ipRecordSid: string;
    /** The SID of the SIP Domain that the IP Record should be mapped to. */
    sipDomainSid: string;
}
/**
 * Options to pass to each
 */
export interface SourceIpMappingListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SourceIpMappingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SourceIpMappingListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SourceIpMappingListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SourceIpMappingContext {
    /**
     * Remove a SourceIpMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SourceIpMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SourceIpMappingInstance
     */
    fetch(callback?: (error: Error | null, item?: SourceIpMappingInstance) => any): Promise<SourceIpMappingInstance>;
    /**
     * Update a SourceIpMappingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SourceIpMappingInstance
     */
    update(params: SourceIpMappingContextUpdateOptions, callback?: (error: Error | null, item?: SourceIpMappingInstance) => any): Promise<SourceIpMappingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SourceIpMappingContextSolution {
    sid: string;
}
export declare class SourceIpMappingContextImpl implements SourceIpMappingContext {
    protected _version: V1;
    protected _solution: SourceIpMappingContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SourceIpMappingInstance) => any): Promise<SourceIpMappingInstance>;
    update(params: SourceIpMappingContextUpdateOptions, callback?: (error: Error | null, item?: SourceIpMappingInstance) => any): Promise<SourceIpMappingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SourceIpMappingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SourceIpMappingPayload extends TwilioResponsePayload {
    source_ip_mappings: SourceIpMappingResource[];
}
interface SourceIpMappingResource {
    sid: string;
    ip_record_sid: string;
    sip_domain_sid: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class SourceIpMappingInstance {
    protected _version: V1;
    protected _solution: SourceIpMappingContextSolution;
    protected _context?: SourceIpMappingContext;
    constructor(_version: V1, payload: SourceIpMappingResource, sid?: string);
    /**
     * The unique string that we created to identify the IP Record resource.
     */
    sid: string;
    /**
     * The Twilio-provided string that uniquely identifies the IP Record resource to map from.
     */
    ipRecordSid: string;
    /**
     * The SID of the SIP Domain that the IP Record is mapped to.
     */
    sipDomainSid: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a SourceIpMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SourceIpMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SourceIpMappingInstance
     */
    fetch(callback?: (error: Error | null, item?: SourceIpMappingInstance) => any): Promise<SourceIpMappingInstance>;
    /**
     * Update a SourceIpMappingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SourceIpMappingInstance
     */
    update(params: SourceIpMappingContextUpdateOptions, callback?: (error: Error | null, item?: SourceIpMappingInstance) => any): Promise<SourceIpMappingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        ipRecordSid: string;
        sipDomainSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SourceIpMappingSolution {
}
export interface SourceIpMappingListInstance {
    _version: V1;
    _solution: SourceIpMappingSolution;
    _uri: string;
    (sid: string): SourceIpMappingContext;
    get(sid: string): SourceIpMappingContext;
    /**
     * Create a SourceIpMappingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SourceIpMappingInstance
     */
    create(params: SourceIpMappingListInstanceCreateOptions, callback?: (error: Error | null, item?: SourceIpMappingInstance) => any): Promise<SourceIpMappingInstance>;
    /**
     * Streams SourceIpMappingInstance records from the API.
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
     * @param { SourceIpMappingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SourceIpMappingInstance, done: (err?: Error) => void) => void): void;
    each(params: SourceIpMappingListInstanceEachOptions, callback?: (item: SourceIpMappingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SourceIpMappingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SourceIpMappingPage) => any): Promise<SourceIpMappingPage>;
    /**
     * Lists SourceIpMappingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SourceIpMappingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SourceIpMappingInstance[]) => any): Promise<SourceIpMappingInstance[]>;
    list(params: SourceIpMappingListInstanceOptions, callback?: (error: Error | null, items: SourceIpMappingInstance[]) => any): Promise<SourceIpMappingInstance[]>;
    /**
     * Retrieve a single page of SourceIpMappingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SourceIpMappingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SourceIpMappingPage) => any): Promise<SourceIpMappingPage>;
    page(params: SourceIpMappingListInstancePageOptions, callback?: (error: Error | null, items: SourceIpMappingPage) => any): Promise<SourceIpMappingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SourceIpMappingListInstance(version: V1): SourceIpMappingListInstance;
export declare class SourceIpMappingPage extends Page<V1, SourceIpMappingPayload, SourceIpMappingResource, SourceIpMappingInstance> {
    /**
     * Initialize the SourceIpMappingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SourceIpMappingSolution);
    /**
     * Build an instance of SourceIpMappingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SourceIpMappingResource): SourceIpMappingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
