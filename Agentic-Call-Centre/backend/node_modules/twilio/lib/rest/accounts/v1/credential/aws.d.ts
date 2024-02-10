/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to update a AwsInstance
 */
export interface AwsContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
}
/**
 * Options to pass to create a AwsInstance
 */
export interface AwsListInstanceCreateOptions {
    /** A string that contains the AWS access credentials in the format `<AWS_ACCESS_KEY_ID>:<AWS_SECRET_ACCESS_KEY>`. For example, `AKIAIOSFODNN7EXAMPLE:wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` */
    credentials: string;
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
    /** The SID of the Subaccount that this Credential should be associated with. Must be a valid Subaccount of the account issuing the request. */
    accountSid?: string;
}
/**
 * Options to pass to each
 */
export interface AwsListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AwsInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AwsListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AwsListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AwsContext {
    /**
     * Remove a AwsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AwsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AwsInstance
     */
    fetch(callback?: (error: Error | null, item?: AwsInstance) => any): Promise<AwsInstance>;
    /**
     * Update a AwsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AwsInstance
     */
    update(callback?: (error: Error | null, item?: AwsInstance) => any): Promise<AwsInstance>;
    /**
     * Update a AwsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AwsInstance
     */
    update(params: AwsContextUpdateOptions, callback?: (error: Error | null, item?: AwsInstance) => any): Promise<AwsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AwsContextSolution {
    sid: string;
}
export declare class AwsContextImpl implements AwsContext {
    protected _version: V1;
    protected _solution: AwsContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AwsInstance) => any): Promise<AwsInstance>;
    update(params?: AwsContextUpdateOptions | ((error: Error | null, item?: AwsInstance) => any), callback?: (error: Error | null, item?: AwsInstance) => any): Promise<AwsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AwsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AwsPayload extends TwilioResponsePayload {
    credentials: AwsResource[];
}
interface AwsResource {
    sid: string;
    account_sid: string;
    friendly_name: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class AwsInstance {
    protected _version: V1;
    protected _solution: AwsContextSolution;
    protected _context?: AwsContext;
    constructor(_version: V1, payload: AwsResource, sid?: string);
    /**
     * The unique string that we created to identify the AWS resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the AWS resource.
     */
    accountSid: string;
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
     * The URI for this resource, relative to `https://accounts.twilio.com`
     */
    url: string;
    private get _proxy();
    /**
     * Remove a AwsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AwsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AwsInstance
     */
    fetch(callback?: (error: Error | null, item?: AwsInstance) => any): Promise<AwsInstance>;
    /**
     * Update a AwsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AwsInstance
     */
    update(callback?: (error: Error | null, item?: AwsInstance) => any): Promise<AwsInstance>;
    /**
     * Update a AwsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AwsInstance
     */
    update(params: AwsContextUpdateOptions, callback?: (error: Error | null, item?: AwsInstance) => any): Promise<AwsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        friendlyName: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AwsSolution {
}
export interface AwsListInstance {
    _version: V1;
    _solution: AwsSolution;
    _uri: string;
    (sid: string): AwsContext;
    get(sid: string): AwsContext;
    /**
     * Create a AwsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AwsInstance
     */
    create(params: AwsListInstanceCreateOptions, callback?: (error: Error | null, item?: AwsInstance) => any): Promise<AwsInstance>;
    /**
     * Streams AwsInstance records from the API.
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
     * @param { AwsListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AwsInstance, done: (err?: Error) => void) => void): void;
    each(params: AwsListInstanceEachOptions, callback?: (item: AwsInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AwsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AwsPage) => any): Promise<AwsPage>;
    /**
     * Lists AwsInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AwsListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AwsInstance[]) => any): Promise<AwsInstance[]>;
    list(params: AwsListInstanceOptions, callback?: (error: Error | null, items: AwsInstance[]) => any): Promise<AwsInstance[]>;
    /**
     * Retrieve a single page of AwsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AwsListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AwsPage) => any): Promise<AwsPage>;
    page(params: AwsListInstancePageOptions, callback?: (error: Error | null, items: AwsPage) => any): Promise<AwsPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AwsListInstance(version: V1): AwsListInstance;
export declare class AwsPage extends Page<V1, AwsPayload, AwsResource, AwsInstance> {
    /**
     * Initialize the AwsPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: AwsSolution);
    /**
     * Build an instance of AwsInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AwsResource): AwsInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
