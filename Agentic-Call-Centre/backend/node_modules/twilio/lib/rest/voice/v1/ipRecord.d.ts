/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to update a IpRecordInstance
 */
export interface IpRecordContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It is not unique and can be up to 255 characters long. */
    friendlyName?: string;
}
/**
 * Options to pass to create a IpRecordInstance
 */
export interface IpRecordListInstanceCreateOptions {
    /** An IP address in dotted decimal notation, IPv4 only. */
    ipAddress: string;
    /** A descriptive string that you create to describe the resource. It is not unique and can be up to 255 characters long. */
    friendlyName?: string;
    /** An integer representing the length of the [CIDR](https://tools.ietf.org/html/rfc4632) prefix to use with this IP address. By default the entire IP address is used, which for IPv4 is value 32. */
    cidrPrefixLength?: number;
}
/**
 * Options to pass to each
 */
export interface IpRecordListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: IpRecordInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface IpRecordListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface IpRecordListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface IpRecordContext {
    /**
     * Remove a IpRecordInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a IpRecordInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpRecordInstance
     */
    fetch(callback?: (error: Error | null, item?: IpRecordInstance) => any): Promise<IpRecordInstance>;
    /**
     * Update a IpRecordInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpRecordInstance
     */
    update(callback?: (error: Error | null, item?: IpRecordInstance) => any): Promise<IpRecordInstance>;
    /**
     * Update a IpRecordInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpRecordInstance
     */
    update(params: IpRecordContextUpdateOptions, callback?: (error: Error | null, item?: IpRecordInstance) => any): Promise<IpRecordInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface IpRecordContextSolution {
    sid: string;
}
export declare class IpRecordContextImpl implements IpRecordContext {
    protected _version: V1;
    protected _solution: IpRecordContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: IpRecordInstance) => any): Promise<IpRecordInstance>;
    update(params?: IpRecordContextUpdateOptions | ((error: Error | null, item?: IpRecordInstance) => any), callback?: (error: Error | null, item?: IpRecordInstance) => any): Promise<IpRecordInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): IpRecordContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface IpRecordPayload extends TwilioResponsePayload {
    ip_records: IpRecordResource[];
}
interface IpRecordResource {
    account_sid: string;
    sid: string;
    friendly_name: string;
    ip_address: string;
    cidr_prefix_length: number;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class IpRecordInstance {
    protected _version: V1;
    protected _solution: IpRecordContextSolution;
    protected _context?: IpRecordContext;
    constructor(_version: V1, payload: IpRecordResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the IP Record resource.
     */
    accountSid: string;
    /**
     * The unique string that we created to identify the IP Record resource.
     */
    sid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * An IP address in dotted decimal notation, IPv4 only.
     */
    ipAddress: string;
    /**
     * An integer representing the length of the [CIDR](https://tools.ietf.org/html/rfc4632) prefix to use with this IP address. By default the entire IP address is used, which for IPv4 is value 32.
     */
    cidrPrefixLength: number;
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
     * Remove a IpRecordInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a IpRecordInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpRecordInstance
     */
    fetch(callback?: (error: Error | null, item?: IpRecordInstance) => any): Promise<IpRecordInstance>;
    /**
     * Update a IpRecordInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpRecordInstance
     */
    update(callback?: (error: Error | null, item?: IpRecordInstance) => any): Promise<IpRecordInstance>;
    /**
     * Update a IpRecordInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpRecordInstance
     */
    update(params: IpRecordContextUpdateOptions, callback?: (error: Error | null, item?: IpRecordInstance) => any): Promise<IpRecordInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        sid: string;
        friendlyName: string;
        ipAddress: string;
        cidrPrefixLength: number;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface IpRecordSolution {
}
export interface IpRecordListInstance {
    _version: V1;
    _solution: IpRecordSolution;
    _uri: string;
    (sid: string): IpRecordContext;
    get(sid: string): IpRecordContext;
    /**
     * Create a IpRecordInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpRecordInstance
     */
    create(params: IpRecordListInstanceCreateOptions, callback?: (error: Error | null, item?: IpRecordInstance) => any): Promise<IpRecordInstance>;
    /**
     * Streams IpRecordInstance records from the API.
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
     * @param { IpRecordListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: IpRecordInstance, done: (err?: Error) => void) => void): void;
    each(params: IpRecordListInstanceEachOptions, callback?: (item: IpRecordInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of IpRecordInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: IpRecordPage) => any): Promise<IpRecordPage>;
    /**
     * Lists IpRecordInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { IpRecordListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: IpRecordInstance[]) => any): Promise<IpRecordInstance[]>;
    list(params: IpRecordListInstanceOptions, callback?: (error: Error | null, items: IpRecordInstance[]) => any): Promise<IpRecordInstance[]>;
    /**
     * Retrieve a single page of IpRecordInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { IpRecordListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: IpRecordPage) => any): Promise<IpRecordPage>;
    page(params: IpRecordListInstancePageOptions, callback?: (error: Error | null, items: IpRecordPage) => any): Promise<IpRecordPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function IpRecordListInstance(version: V1): IpRecordListInstance;
export declare class IpRecordPage extends Page<V1, IpRecordPayload, IpRecordResource, IpRecordInstance> {
    /**
     * Initialize the IpRecordPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: IpRecordSolution);
    /**
     * Build an instance of IpRecordInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: IpRecordResource): IpRecordInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
