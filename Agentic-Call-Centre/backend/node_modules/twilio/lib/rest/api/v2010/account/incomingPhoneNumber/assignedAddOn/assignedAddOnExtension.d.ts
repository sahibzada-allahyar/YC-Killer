/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../../base/Page";
import Response from "../../../../../../http/response";
import V2010 from "../../../../V2010";
/**
 * Options to pass to each
 */
export interface AssignedAddOnExtensionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AssignedAddOnExtensionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AssignedAddOnExtensionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AssignedAddOnExtensionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AssignedAddOnExtensionContext {
    /**
     * Fetch a AssignedAddOnExtensionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssignedAddOnExtensionInstance
     */
    fetch(callback?: (error: Error | null, item?: AssignedAddOnExtensionInstance) => any): Promise<AssignedAddOnExtensionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AssignedAddOnExtensionContextSolution {
    accountSid: string;
    resourceSid: string;
    assignedAddOnSid: string;
    sid: string;
}
export declare class AssignedAddOnExtensionContextImpl implements AssignedAddOnExtensionContext {
    protected _version: V2010;
    protected _solution: AssignedAddOnExtensionContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, resourceSid: string, assignedAddOnSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: AssignedAddOnExtensionInstance) => any): Promise<AssignedAddOnExtensionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AssignedAddOnExtensionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AssignedAddOnExtensionPayload extends TwilioResponsePayload {
    extensions: AssignedAddOnExtensionResource[];
}
interface AssignedAddOnExtensionResource {
    sid: string;
    account_sid: string;
    resource_sid: string;
    assigned_add_on_sid: string;
    friendly_name: string;
    product_name: string;
    unique_name: string;
    uri: string;
    enabled: boolean;
}
export declare class AssignedAddOnExtensionInstance {
    protected _version: V2010;
    protected _solution: AssignedAddOnExtensionContextSolution;
    protected _context?: AssignedAddOnExtensionContext;
    constructor(_version: V2010, payload: AssignedAddOnExtensionResource, accountSid: string, resourceSid: string, assignedAddOnSid: string, sid?: string);
    /**
     * The unique string that that we created to identify the resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the resource.
     */
    accountSid: string;
    /**
     * The SID of the Phone Number to which the Add-on is assigned.
     */
    resourceSid: string;
    /**
     * The SID that uniquely identifies the assigned Add-on installation.
     */
    assignedAddOnSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * A string that you assigned to describe the Product this Extension is used within.
     */
    productName: string;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used in place of the resource\'s `sid` in the URL to address the resource.
     */
    uniqueName: string;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    /**
     * Whether the Extension will be invoked.
     */
    enabled: boolean;
    private get _proxy();
    /**
     * Fetch a AssignedAddOnExtensionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssignedAddOnExtensionInstance
     */
    fetch(callback?: (error: Error | null, item?: AssignedAddOnExtensionInstance) => any): Promise<AssignedAddOnExtensionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        resourceSid: string;
        assignedAddOnSid: string;
        friendlyName: string;
        productName: string;
        uniqueName: string;
        uri: string;
        enabled: boolean;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AssignedAddOnExtensionSolution {
    accountSid: string;
    resourceSid: string;
    assignedAddOnSid: string;
}
export interface AssignedAddOnExtensionListInstance {
    _version: V2010;
    _solution: AssignedAddOnExtensionSolution;
    _uri: string;
    (sid: string): AssignedAddOnExtensionContext;
    get(sid: string): AssignedAddOnExtensionContext;
    /**
     * Streams AssignedAddOnExtensionInstance records from the API.
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
     * @param { AssignedAddOnExtensionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AssignedAddOnExtensionInstance, done: (err?: Error) => void) => void): void;
    each(params: AssignedAddOnExtensionListInstanceEachOptions, callback?: (item: AssignedAddOnExtensionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AssignedAddOnExtensionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AssignedAddOnExtensionPage) => any): Promise<AssignedAddOnExtensionPage>;
    /**
     * Lists AssignedAddOnExtensionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AssignedAddOnExtensionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AssignedAddOnExtensionInstance[]) => any): Promise<AssignedAddOnExtensionInstance[]>;
    list(params: AssignedAddOnExtensionListInstanceOptions, callback?: (error: Error | null, items: AssignedAddOnExtensionInstance[]) => any): Promise<AssignedAddOnExtensionInstance[]>;
    /**
     * Retrieve a single page of AssignedAddOnExtensionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AssignedAddOnExtensionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AssignedAddOnExtensionPage) => any): Promise<AssignedAddOnExtensionPage>;
    page(params: AssignedAddOnExtensionListInstancePageOptions, callback?: (error: Error | null, items: AssignedAddOnExtensionPage) => any): Promise<AssignedAddOnExtensionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AssignedAddOnExtensionListInstance(version: V2010, accountSid: string, resourceSid: string, assignedAddOnSid: string): AssignedAddOnExtensionListInstance;
export declare class AssignedAddOnExtensionPage extends Page<V2010, AssignedAddOnExtensionPayload, AssignedAddOnExtensionResource, AssignedAddOnExtensionInstance> {
    /**
     * Initialize the AssignedAddOnExtensionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: AssignedAddOnExtensionSolution);
    /**
     * Build an instance of AssignedAddOnExtensionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AssignedAddOnExtensionResource): AssignedAddOnExtensionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
