/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to create a CustomerProfilesChannelEndpointAssignmentInstance
 */
export interface CustomerProfilesChannelEndpointAssignmentListInstanceCreateOptions {
    /** The type of channel endpoint. eg: phone-number */
    channelEndpointType: string;
    /** The SID of an channel endpoint */
    channelEndpointSid: string;
}
/**
 * Options to pass to each
 */
export interface CustomerProfilesChannelEndpointAssignmentListInstanceEachOptions {
    /** The SID of an channel endpoint */
    channelEndpointSid?: string;
    /** comma separated list of channel endpoint sids */
    channelEndpointSids?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: CustomerProfilesChannelEndpointAssignmentInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface CustomerProfilesChannelEndpointAssignmentListInstanceOptions {
    /** The SID of an channel endpoint */
    channelEndpointSid?: string;
    /** comma separated list of channel endpoint sids */
    channelEndpointSids?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CustomerProfilesChannelEndpointAssignmentListInstancePageOptions {
    /** The SID of an channel endpoint */
    channelEndpointSid?: string;
    /** comma separated list of channel endpoint sids */
    channelEndpointSids?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CustomerProfilesChannelEndpointAssignmentContext {
    /**
     * Remove a CustomerProfilesChannelEndpointAssignmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CustomerProfilesChannelEndpointAssignmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesChannelEndpointAssignmentInstance
     */
    fetch(callback?: (error: Error | null, item?: CustomerProfilesChannelEndpointAssignmentInstance) => any): Promise<CustomerProfilesChannelEndpointAssignmentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface CustomerProfilesChannelEndpointAssignmentContextSolution {
    customerProfileSid: string;
    sid: string;
}
export declare class CustomerProfilesChannelEndpointAssignmentContextImpl implements CustomerProfilesChannelEndpointAssignmentContext {
    protected _version: V1;
    protected _solution: CustomerProfilesChannelEndpointAssignmentContextSolution;
    protected _uri: string;
    constructor(_version: V1, customerProfileSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: CustomerProfilesChannelEndpointAssignmentInstance) => any): Promise<CustomerProfilesChannelEndpointAssignmentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): CustomerProfilesChannelEndpointAssignmentContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface CustomerProfilesChannelEndpointAssignmentPayload extends TwilioResponsePayload {
    results: CustomerProfilesChannelEndpointAssignmentResource[];
}
interface CustomerProfilesChannelEndpointAssignmentResource {
    sid: string;
    customer_profile_sid: string;
    account_sid: string;
    channel_endpoint_type: string;
    channel_endpoint_sid: string;
    date_created: Date;
    url: string;
}
export declare class CustomerProfilesChannelEndpointAssignmentInstance {
    protected _version: V1;
    protected _solution: CustomerProfilesChannelEndpointAssignmentContextSolution;
    protected _context?: CustomerProfilesChannelEndpointAssignmentContext;
    constructor(_version: V1, payload: CustomerProfilesChannelEndpointAssignmentResource, customerProfileSid: string, sid?: string);
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
     * The type of channel endpoint. eg: phone-number
     */
    channelEndpointType: string;
    /**
     * The SID of an channel endpoint
     */
    channelEndpointSid: string;
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
     * Remove a CustomerProfilesChannelEndpointAssignmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CustomerProfilesChannelEndpointAssignmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesChannelEndpointAssignmentInstance
     */
    fetch(callback?: (error: Error | null, item?: CustomerProfilesChannelEndpointAssignmentInstance) => any): Promise<CustomerProfilesChannelEndpointAssignmentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        customerProfileSid: string;
        accountSid: string;
        channelEndpointType: string;
        channelEndpointSid: string;
        dateCreated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CustomerProfilesChannelEndpointAssignmentSolution {
    customerProfileSid: string;
}
export interface CustomerProfilesChannelEndpointAssignmentListInstance {
    _version: V1;
    _solution: CustomerProfilesChannelEndpointAssignmentSolution;
    _uri: string;
    (sid: string): CustomerProfilesChannelEndpointAssignmentContext;
    get(sid: string): CustomerProfilesChannelEndpointAssignmentContext;
    /**
     * Create a CustomerProfilesChannelEndpointAssignmentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesChannelEndpointAssignmentInstance
     */
    create(params: CustomerProfilesChannelEndpointAssignmentListInstanceCreateOptions, callback?: (error: Error | null, item?: CustomerProfilesChannelEndpointAssignmentInstance) => any): Promise<CustomerProfilesChannelEndpointAssignmentInstance>;
    /**
     * Streams CustomerProfilesChannelEndpointAssignmentInstance records from the API.
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
     * @param { CustomerProfilesChannelEndpointAssignmentListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: CustomerProfilesChannelEndpointAssignmentInstance, done: (err?: Error) => void) => void): void;
    each(params: CustomerProfilesChannelEndpointAssignmentListInstanceEachOptions, callback?: (item: CustomerProfilesChannelEndpointAssignmentInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of CustomerProfilesChannelEndpointAssignmentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: CustomerProfilesChannelEndpointAssignmentPage) => any): Promise<CustomerProfilesChannelEndpointAssignmentPage>;
    /**
     * Lists CustomerProfilesChannelEndpointAssignmentInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CustomerProfilesChannelEndpointAssignmentListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: CustomerProfilesChannelEndpointAssignmentInstance[]) => any): Promise<CustomerProfilesChannelEndpointAssignmentInstance[]>;
    list(params: CustomerProfilesChannelEndpointAssignmentListInstanceOptions, callback?: (error: Error | null, items: CustomerProfilesChannelEndpointAssignmentInstance[]) => any): Promise<CustomerProfilesChannelEndpointAssignmentInstance[]>;
    /**
     * Retrieve a single page of CustomerProfilesChannelEndpointAssignmentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CustomerProfilesChannelEndpointAssignmentListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: CustomerProfilesChannelEndpointAssignmentPage) => any): Promise<CustomerProfilesChannelEndpointAssignmentPage>;
    page(params: CustomerProfilesChannelEndpointAssignmentListInstancePageOptions, callback?: (error: Error | null, items: CustomerProfilesChannelEndpointAssignmentPage) => any): Promise<CustomerProfilesChannelEndpointAssignmentPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CustomerProfilesChannelEndpointAssignmentListInstance(version: V1, customerProfileSid: string): CustomerProfilesChannelEndpointAssignmentListInstance;
export declare class CustomerProfilesChannelEndpointAssignmentPage extends Page<V1, CustomerProfilesChannelEndpointAssignmentPayload, CustomerProfilesChannelEndpointAssignmentResource, CustomerProfilesChannelEndpointAssignmentInstance> {
    /**
     * Initialize the CustomerProfilesChannelEndpointAssignmentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: CustomerProfilesChannelEndpointAssignmentSolution);
    /**
     * Build an instance of CustomerProfilesChannelEndpointAssignmentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CustomerProfilesChannelEndpointAssignmentResource): CustomerProfilesChannelEndpointAssignmentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
