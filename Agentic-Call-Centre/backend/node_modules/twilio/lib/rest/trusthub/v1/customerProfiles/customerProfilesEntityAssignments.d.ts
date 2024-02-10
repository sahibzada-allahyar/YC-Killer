/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to create a CustomerProfilesEntityAssignmentsInstance
 */
export interface CustomerProfilesEntityAssignmentsListInstanceCreateOptions {
    /** The SID of an object bag that holds information of the different items. */
    objectSid: string;
}
/**
 * Options to pass to each
 */
export interface CustomerProfilesEntityAssignmentsListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: CustomerProfilesEntityAssignmentsInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface CustomerProfilesEntityAssignmentsListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CustomerProfilesEntityAssignmentsListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CustomerProfilesEntityAssignmentsContext {
    /**
     * Remove a CustomerProfilesEntityAssignmentsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CustomerProfilesEntityAssignmentsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesEntityAssignmentsInstance
     */
    fetch(callback?: (error: Error | null, item?: CustomerProfilesEntityAssignmentsInstance) => any): Promise<CustomerProfilesEntityAssignmentsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface CustomerProfilesEntityAssignmentsContextSolution {
    customerProfileSid: string;
    sid: string;
}
export declare class CustomerProfilesEntityAssignmentsContextImpl implements CustomerProfilesEntityAssignmentsContext {
    protected _version: V1;
    protected _solution: CustomerProfilesEntityAssignmentsContextSolution;
    protected _uri: string;
    constructor(_version: V1, customerProfileSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: CustomerProfilesEntityAssignmentsInstance) => any): Promise<CustomerProfilesEntityAssignmentsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): CustomerProfilesEntityAssignmentsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface CustomerProfilesEntityAssignmentsPayload extends TwilioResponsePayload {
    results: CustomerProfilesEntityAssignmentsResource[];
}
interface CustomerProfilesEntityAssignmentsResource {
    sid: string;
    customer_profile_sid: string;
    account_sid: string;
    object_sid: string;
    date_created: Date;
    url: string;
}
export declare class CustomerProfilesEntityAssignmentsInstance {
    protected _version: V1;
    protected _solution: CustomerProfilesEntityAssignmentsContextSolution;
    protected _context?: CustomerProfilesEntityAssignmentsContext;
    constructor(_version: V1, payload: CustomerProfilesEntityAssignmentsResource, customerProfileSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Item Assignment resource.
     */
    sid: string;
    /**
     * The unique string that we created to identify the CustomerProfile resource.
     */
    customerProfileSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Item Assignment resource.
     */
    accountSid: string;
    /**
     * The SID of an object bag that holds information of the different items.
     */
    objectSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The absolute URL of the Identity resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a CustomerProfilesEntityAssignmentsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CustomerProfilesEntityAssignmentsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesEntityAssignmentsInstance
     */
    fetch(callback?: (error: Error | null, item?: CustomerProfilesEntityAssignmentsInstance) => any): Promise<CustomerProfilesEntityAssignmentsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        customerProfileSid: string;
        accountSid: string;
        objectSid: string;
        dateCreated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CustomerProfilesEntityAssignmentsSolution {
    customerProfileSid: string;
}
export interface CustomerProfilesEntityAssignmentsListInstance {
    _version: V1;
    _solution: CustomerProfilesEntityAssignmentsSolution;
    _uri: string;
    (sid: string): CustomerProfilesEntityAssignmentsContext;
    get(sid: string): CustomerProfilesEntityAssignmentsContext;
    /**
     * Create a CustomerProfilesEntityAssignmentsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesEntityAssignmentsInstance
     */
    create(params: CustomerProfilesEntityAssignmentsListInstanceCreateOptions, callback?: (error: Error | null, item?: CustomerProfilesEntityAssignmentsInstance) => any): Promise<CustomerProfilesEntityAssignmentsInstance>;
    /**
     * Streams CustomerProfilesEntityAssignmentsInstance records from the API.
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
     * @param { CustomerProfilesEntityAssignmentsListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: CustomerProfilesEntityAssignmentsInstance, done: (err?: Error) => void) => void): void;
    each(params: CustomerProfilesEntityAssignmentsListInstanceEachOptions, callback?: (item: CustomerProfilesEntityAssignmentsInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of CustomerProfilesEntityAssignmentsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: CustomerProfilesEntityAssignmentsPage) => any): Promise<CustomerProfilesEntityAssignmentsPage>;
    /**
     * Lists CustomerProfilesEntityAssignmentsInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CustomerProfilesEntityAssignmentsListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: CustomerProfilesEntityAssignmentsInstance[]) => any): Promise<CustomerProfilesEntityAssignmentsInstance[]>;
    list(params: CustomerProfilesEntityAssignmentsListInstanceOptions, callback?: (error: Error | null, items: CustomerProfilesEntityAssignmentsInstance[]) => any): Promise<CustomerProfilesEntityAssignmentsInstance[]>;
    /**
     * Retrieve a single page of CustomerProfilesEntityAssignmentsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CustomerProfilesEntityAssignmentsListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: CustomerProfilesEntityAssignmentsPage) => any): Promise<CustomerProfilesEntityAssignmentsPage>;
    page(params: CustomerProfilesEntityAssignmentsListInstancePageOptions, callback?: (error: Error | null, items: CustomerProfilesEntityAssignmentsPage) => any): Promise<CustomerProfilesEntityAssignmentsPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CustomerProfilesEntityAssignmentsListInstance(version: V1, customerProfileSid: string): CustomerProfilesEntityAssignmentsListInstance;
export declare class CustomerProfilesEntityAssignmentsPage extends Page<V1, CustomerProfilesEntityAssignmentsPayload, CustomerProfilesEntityAssignmentsResource, CustomerProfilesEntityAssignmentsInstance> {
    /**
     * Initialize the CustomerProfilesEntityAssignmentsPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: CustomerProfilesEntityAssignmentsSolution);
    /**
     * Build an instance of CustomerProfilesEntityAssignmentsInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CustomerProfilesEntityAssignmentsResource): CustomerProfilesEntityAssignmentsInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
