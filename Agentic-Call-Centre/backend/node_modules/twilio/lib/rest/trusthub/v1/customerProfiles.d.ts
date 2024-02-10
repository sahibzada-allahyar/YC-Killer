/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { CustomerProfilesChannelEndpointAssignmentListInstance } from "./customerProfiles/customerProfilesChannelEndpointAssignment";
import { CustomerProfilesEntityAssignmentsListInstance } from "./customerProfiles/customerProfilesEntityAssignments";
import { CustomerProfilesEvaluationsListInstance } from "./customerProfiles/customerProfilesEvaluations";
export type CustomerProfilesStatus = "draft" | "pending-review" | "in-review" | "twilio-rejected" | "twilio-approved";
/**
 * Options to pass to update a CustomerProfilesInstance
 */
export interface CustomerProfilesContextUpdateOptions {
    /**  */
    status?: CustomerProfilesStatus;
    /** The URL we call to inform your application of status changes. */
    statusCallback?: string;
    /** The string that you assigned to describe the resource. */
    friendlyName?: string;
    /** The email address that will receive updates when the Customer-Profile resource changes status. */
    email?: string;
}
/**
 * Options to pass to create a CustomerProfilesInstance
 */
export interface CustomerProfilesListInstanceCreateOptions {
    /** The string that you assigned to describe the resource. */
    friendlyName: string;
    /** The email address that will receive updates when the Customer-Profile resource changes status. */
    email: string;
    /** The unique string of a policy that is associated to the Customer-Profile resource. */
    policySid: string;
    /** The URL we call to inform your application of status changes. */
    statusCallback?: string;
}
/**
 * Options to pass to each
 */
export interface CustomerProfilesListInstanceEachOptions {
    /** The verification status of the Customer-Profile resource. */
    status?: CustomerProfilesStatus;
    /** The string that you assigned to describe the resource. */
    friendlyName?: string;
    /** The unique string of a policy that is associated to the Customer-Profile resource. */
    policySid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: CustomerProfilesInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface CustomerProfilesListInstanceOptions {
    /** The verification status of the Customer-Profile resource. */
    status?: CustomerProfilesStatus;
    /** The string that you assigned to describe the resource. */
    friendlyName?: string;
    /** The unique string of a policy that is associated to the Customer-Profile resource. */
    policySid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CustomerProfilesListInstancePageOptions {
    /** The verification status of the Customer-Profile resource. */
    status?: CustomerProfilesStatus;
    /** The string that you assigned to describe the resource. */
    friendlyName?: string;
    /** The unique string of a policy that is associated to the Customer-Profile resource. */
    policySid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CustomerProfilesContext {
    customerProfilesChannelEndpointAssignment: CustomerProfilesChannelEndpointAssignmentListInstance;
    customerProfilesEntityAssignments: CustomerProfilesEntityAssignmentsListInstance;
    customerProfilesEvaluations: CustomerProfilesEvaluationsListInstance;
    /**
     * Remove a CustomerProfilesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CustomerProfilesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesInstance
     */
    fetch(callback?: (error: Error | null, item?: CustomerProfilesInstance) => any): Promise<CustomerProfilesInstance>;
    /**
     * Update a CustomerProfilesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesInstance
     */
    update(callback?: (error: Error | null, item?: CustomerProfilesInstance) => any): Promise<CustomerProfilesInstance>;
    /**
     * Update a CustomerProfilesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesInstance
     */
    update(params: CustomerProfilesContextUpdateOptions, callback?: (error: Error | null, item?: CustomerProfilesInstance) => any): Promise<CustomerProfilesInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface CustomerProfilesContextSolution {
    sid: string;
}
export declare class CustomerProfilesContextImpl implements CustomerProfilesContext {
    protected _version: V1;
    protected _solution: CustomerProfilesContextSolution;
    protected _uri: string;
    protected _customerProfilesChannelEndpointAssignment?: CustomerProfilesChannelEndpointAssignmentListInstance;
    protected _customerProfilesEntityAssignments?: CustomerProfilesEntityAssignmentsListInstance;
    protected _customerProfilesEvaluations?: CustomerProfilesEvaluationsListInstance;
    constructor(_version: V1, sid: string);
    get customerProfilesChannelEndpointAssignment(): CustomerProfilesChannelEndpointAssignmentListInstance;
    get customerProfilesEntityAssignments(): CustomerProfilesEntityAssignmentsListInstance;
    get customerProfilesEvaluations(): CustomerProfilesEvaluationsListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: CustomerProfilesInstance) => any): Promise<CustomerProfilesInstance>;
    update(params?: CustomerProfilesContextUpdateOptions | ((error: Error | null, item?: CustomerProfilesInstance) => any), callback?: (error: Error | null, item?: CustomerProfilesInstance) => any): Promise<CustomerProfilesInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): CustomerProfilesContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface CustomerProfilesPayload extends TwilioResponsePayload {
    results: CustomerProfilesResource[];
}
interface CustomerProfilesResource {
    sid: string;
    account_sid: string;
    policy_sid: string;
    friendly_name: string;
    status: CustomerProfilesStatus;
    valid_until: Date;
    email: string;
    status_callback: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class CustomerProfilesInstance {
    protected _version: V1;
    protected _solution: CustomerProfilesContextSolution;
    protected _context?: CustomerProfilesContext;
    constructor(_version: V1, payload: CustomerProfilesResource, sid?: string);
    /**
     * The unique string that we created to identify the Customer-Profile resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Customer-Profile resource.
     */
    accountSid: string;
    /**
     * The unique string of a policy that is associated to the Customer-Profile resource.
     */
    policySid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    status: CustomerProfilesStatus;
    /**
     * The date and time in GMT in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format when the resource will be valid until.
     */
    validUntil: Date;
    /**
     * The email address that will receive updates when the Customer-Profile resource changes status.
     */
    email: string;
    /**
     * The URL we call to inform your application of status changes.
     */
    statusCallback: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Customer-Profile resource.
     */
    url: string;
    /**
     * The URLs of the Assigned Items of the Customer-Profile resource.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a CustomerProfilesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CustomerProfilesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesInstance
     */
    fetch(callback?: (error: Error | null, item?: CustomerProfilesInstance) => any): Promise<CustomerProfilesInstance>;
    /**
     * Update a CustomerProfilesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesInstance
     */
    update(callback?: (error: Error | null, item?: CustomerProfilesInstance) => any): Promise<CustomerProfilesInstance>;
    /**
     * Update a CustomerProfilesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesInstance
     */
    update(params: CustomerProfilesContextUpdateOptions, callback?: (error: Error | null, item?: CustomerProfilesInstance) => any): Promise<CustomerProfilesInstance>;
    /**
     * Access the customerProfilesChannelEndpointAssignment.
     */
    customerProfilesChannelEndpointAssignment(): CustomerProfilesChannelEndpointAssignmentListInstance;
    /**
     * Access the customerProfilesEntityAssignments.
     */
    customerProfilesEntityAssignments(): CustomerProfilesEntityAssignmentsListInstance;
    /**
     * Access the customerProfilesEvaluations.
     */
    customerProfilesEvaluations(): CustomerProfilesEvaluationsListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        policySid: string;
        friendlyName: string;
        status: CustomerProfilesStatus;
        validUntil: Date;
        email: string;
        statusCallback: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CustomerProfilesSolution {
}
export interface CustomerProfilesListInstance {
    _version: V1;
    _solution: CustomerProfilesSolution;
    _uri: string;
    (sid: string): CustomerProfilesContext;
    get(sid: string): CustomerProfilesContext;
    /**
     * Create a CustomerProfilesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CustomerProfilesInstance
     */
    create(params: CustomerProfilesListInstanceCreateOptions, callback?: (error: Error | null, item?: CustomerProfilesInstance) => any): Promise<CustomerProfilesInstance>;
    /**
     * Streams CustomerProfilesInstance records from the API.
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
     * @param { CustomerProfilesListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: CustomerProfilesInstance, done: (err?: Error) => void) => void): void;
    each(params: CustomerProfilesListInstanceEachOptions, callback?: (item: CustomerProfilesInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of CustomerProfilesInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: CustomerProfilesPage) => any): Promise<CustomerProfilesPage>;
    /**
     * Lists CustomerProfilesInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CustomerProfilesListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: CustomerProfilesInstance[]) => any): Promise<CustomerProfilesInstance[]>;
    list(params: CustomerProfilesListInstanceOptions, callback?: (error: Error | null, items: CustomerProfilesInstance[]) => any): Promise<CustomerProfilesInstance[]>;
    /**
     * Retrieve a single page of CustomerProfilesInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CustomerProfilesListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: CustomerProfilesPage) => any): Promise<CustomerProfilesPage>;
    page(params: CustomerProfilesListInstancePageOptions, callback?: (error: Error | null, items: CustomerProfilesPage) => any): Promise<CustomerProfilesPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CustomerProfilesListInstance(version: V1): CustomerProfilesListInstance;
export declare class CustomerProfilesPage extends Page<V1, CustomerProfilesPayload, CustomerProfilesResource, CustomerProfilesInstance> {
    /**
     * Initialize the CustomerProfilesPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: CustomerProfilesSolution);
    /**
     * Build an instance of CustomerProfilesInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CustomerProfilesResource): CustomerProfilesInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
