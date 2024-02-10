/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../../base/Page";
import Response from "../../../../../../http/response";
import V2010 from "../../../../V2010";
/**
 * Options to pass to create a IpAccessControlListMappingInstance
 */
export interface IpAccessControlListMappingListInstanceCreateOptions {
    /** The unique id of the IP access control list to map to the SIP domain. */
    ipAccessControlListSid: string;
}
/**
 * Options to pass to each
 */
export interface IpAccessControlListMappingListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: IpAccessControlListMappingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface IpAccessControlListMappingListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface IpAccessControlListMappingListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface IpAccessControlListMappingContext {
    /**
     * Remove a IpAccessControlListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a IpAccessControlListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpAccessControlListMappingInstance
     */
    fetch(callback?: (error: Error | null, item?: IpAccessControlListMappingInstance) => any): Promise<IpAccessControlListMappingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface IpAccessControlListMappingContextSolution {
    accountSid: string;
    domainSid: string;
    sid: string;
}
export declare class IpAccessControlListMappingContextImpl implements IpAccessControlListMappingContext {
    protected _version: V2010;
    protected _solution: IpAccessControlListMappingContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, domainSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: IpAccessControlListMappingInstance) => any): Promise<IpAccessControlListMappingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): IpAccessControlListMappingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface IpAccessControlListMappingPayload extends TwilioResponsePayload {
    ip_access_control_list_mappings: IpAccessControlListMappingResource[];
}
interface IpAccessControlListMappingResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    domain_sid: string;
    friendly_name: string;
    sid: string;
    uri: string;
}
export declare class IpAccessControlListMappingInstance {
    protected _version: V2010;
    protected _solution: IpAccessControlListMappingContextSolution;
    protected _context?: IpAccessControlListMappingContext;
    constructor(_version: V2010, payload: IpAccessControlListMappingResource, accountSid: string, domainSid: string, sid?: string);
    /**
     * The unique id of the Account that is responsible for this resource.
     */
    accountSid: string;
    /**
     * The date that this resource was created, given as GMT in [RFC 2822](https://www.php.net/manual/en/class.datetime.php#datetime.constants.rfc2822) format.
     */
    dateCreated: Date;
    /**
     * The date that this resource was last updated, given as GMT in [RFC 2822](https://www.php.net/manual/en/class.datetime.php#datetime.constants.rfc2822) format.
     */
    dateUpdated: Date;
    /**
     * The unique string that is created to identify the SipDomain resource.
     */
    domainSid: string;
    /**
     * A human readable descriptive text for this resource, up to 64 characters long.
     */
    friendlyName: string;
    /**
     * A 34 character string that uniquely identifies this resource.
     */
    sid: string;
    /**
     * The URI for this resource, relative to `https://api.twilio.com`
     */
    uri: string;
    private get _proxy();
    /**
     * Remove a IpAccessControlListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a IpAccessControlListMappingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpAccessControlListMappingInstance
     */
    fetch(callback?: (error: Error | null, item?: IpAccessControlListMappingInstance) => any): Promise<IpAccessControlListMappingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        domainSid: string;
        friendlyName: string;
        sid: string;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface IpAccessControlListMappingSolution {
    accountSid: string;
    domainSid: string;
}
export interface IpAccessControlListMappingListInstance {
    _version: V2010;
    _solution: IpAccessControlListMappingSolution;
    _uri: string;
    (sid: string): IpAccessControlListMappingContext;
    get(sid: string): IpAccessControlListMappingContext;
    /**
     * Create a IpAccessControlListMappingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpAccessControlListMappingInstance
     */
    create(params: IpAccessControlListMappingListInstanceCreateOptions, callback?: (error: Error | null, item?: IpAccessControlListMappingInstance) => any): Promise<IpAccessControlListMappingInstance>;
    /**
     * Streams IpAccessControlListMappingInstance records from the API.
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
     * @param { IpAccessControlListMappingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: IpAccessControlListMappingInstance, done: (err?: Error) => void) => void): void;
    each(params: IpAccessControlListMappingListInstanceEachOptions, callback?: (item: IpAccessControlListMappingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of IpAccessControlListMappingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: IpAccessControlListMappingPage) => any): Promise<IpAccessControlListMappingPage>;
    /**
     * Lists IpAccessControlListMappingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { IpAccessControlListMappingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: IpAccessControlListMappingInstance[]) => any): Promise<IpAccessControlListMappingInstance[]>;
    list(params: IpAccessControlListMappingListInstanceOptions, callback?: (error: Error | null, items: IpAccessControlListMappingInstance[]) => any): Promise<IpAccessControlListMappingInstance[]>;
    /**
     * Retrieve a single page of IpAccessControlListMappingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { IpAccessControlListMappingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: IpAccessControlListMappingPage) => any): Promise<IpAccessControlListMappingPage>;
    page(params: IpAccessControlListMappingListInstancePageOptions, callback?: (error: Error | null, items: IpAccessControlListMappingPage) => any): Promise<IpAccessControlListMappingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function IpAccessControlListMappingListInstance(version: V2010, accountSid: string, domainSid: string): IpAccessControlListMappingListInstance;
export declare class IpAccessControlListMappingPage extends Page<V2010, IpAccessControlListMappingPayload, IpAccessControlListMappingResource, IpAccessControlListMappingInstance> {
    /**
     * Initialize the IpAccessControlListMappingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: IpAccessControlListMappingSolution);
    /**
     * Build an instance of IpAccessControlListMappingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: IpAccessControlListMappingResource): IpAccessControlListMappingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
