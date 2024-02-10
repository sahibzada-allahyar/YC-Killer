/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to create a IpAccessControlListInstance
 */
export interface IpAccessControlListListInstanceCreateOptions {
    /** The SID of the [IP Access Control List](https://www.twilio.com/docs/voice/sip/api/sip-ipaccesscontrollist-resource) that you want to associate with the trunk. */
    ipAccessControlListSid: string;
}
/**
 * Options to pass to each
 */
export interface IpAccessControlListListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: IpAccessControlListInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface IpAccessControlListListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface IpAccessControlListListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface IpAccessControlListContext {
    /**
     * Remove a IpAccessControlListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a IpAccessControlListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpAccessControlListInstance
     */
    fetch(callback?: (error: Error | null, item?: IpAccessControlListInstance) => any): Promise<IpAccessControlListInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface IpAccessControlListContextSolution {
    trunkSid: string;
    sid: string;
}
export declare class IpAccessControlListContextImpl implements IpAccessControlListContext {
    protected _version: V1;
    protected _solution: IpAccessControlListContextSolution;
    protected _uri: string;
    constructor(_version: V1, trunkSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: IpAccessControlListInstance) => any): Promise<IpAccessControlListInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): IpAccessControlListContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface IpAccessControlListPayload extends TwilioResponsePayload {
    ip_access_control_lists: IpAccessControlListResource[];
}
interface IpAccessControlListResource {
    account_sid: string;
    sid: string;
    trunk_sid: string;
    friendly_name: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class IpAccessControlListInstance {
    protected _version: V1;
    protected _solution: IpAccessControlListContextSolution;
    protected _context?: IpAccessControlListContext;
    constructor(_version: V1, payload: IpAccessControlListResource, trunkSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the IpAccessControlList resource.
     */
    accountSid: string;
    /**
     * The unique string that we created to identify the IpAccessControlList resource.
     */
    sid: string;
    /**
     * The SID of the Trunk the resource is associated with.
     */
    trunkSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a IpAccessControlListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a IpAccessControlListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpAccessControlListInstance
     */
    fetch(callback?: (error: Error | null, item?: IpAccessControlListInstance) => any): Promise<IpAccessControlListInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        sid: string;
        trunkSid: string;
        friendlyName: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface IpAccessControlListSolution {
    trunkSid: string;
}
export interface IpAccessControlListListInstance {
    _version: V1;
    _solution: IpAccessControlListSolution;
    _uri: string;
    (sid: string): IpAccessControlListContext;
    get(sid: string): IpAccessControlListContext;
    /**
     * Create a IpAccessControlListInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed IpAccessControlListInstance
     */
    create(params: IpAccessControlListListInstanceCreateOptions, callback?: (error: Error | null, item?: IpAccessControlListInstance) => any): Promise<IpAccessControlListInstance>;
    /**
     * Streams IpAccessControlListInstance records from the API.
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
     * @param { IpAccessControlListListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: IpAccessControlListInstance, done: (err?: Error) => void) => void): void;
    each(params: IpAccessControlListListInstanceEachOptions, callback?: (item: IpAccessControlListInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of IpAccessControlListInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: IpAccessControlListPage) => any): Promise<IpAccessControlListPage>;
    /**
     * Lists IpAccessControlListInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { IpAccessControlListListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: IpAccessControlListInstance[]) => any): Promise<IpAccessControlListInstance[]>;
    list(params: IpAccessControlListListInstanceOptions, callback?: (error: Error | null, items: IpAccessControlListInstance[]) => any): Promise<IpAccessControlListInstance[]>;
    /**
     * Retrieve a single page of IpAccessControlListInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { IpAccessControlListListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: IpAccessControlListPage) => any): Promise<IpAccessControlListPage>;
    page(params: IpAccessControlListListInstancePageOptions, callback?: (error: Error | null, items: IpAccessControlListPage) => any): Promise<IpAccessControlListPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function IpAccessControlListListInstance(version: V1, trunkSid: string): IpAccessControlListListInstance;
export declare class IpAccessControlListPage extends Page<V1, IpAccessControlListPayload, IpAccessControlListResource, IpAccessControlListInstance> {
    /**
     * Initialize the IpAccessControlListPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: IpAccessControlListSolution);
    /**
     * Build an instance of IpAccessControlListInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: IpAccessControlListResource): IpAccessControlListInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
