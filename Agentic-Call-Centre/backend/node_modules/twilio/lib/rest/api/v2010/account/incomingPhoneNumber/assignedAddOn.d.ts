/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2010 from "../../../V2010";
import { AssignedAddOnExtensionListInstance } from "./assignedAddOn/assignedAddOnExtension";
/**
 * Options to pass to create a AssignedAddOnInstance
 */
export interface AssignedAddOnListInstanceCreateOptions {
    /** The SID that identifies the Add-on installation. */
    installedAddOnSid: string;
}
/**
 * Options to pass to each
 */
export interface AssignedAddOnListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AssignedAddOnInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AssignedAddOnListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AssignedAddOnListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AssignedAddOnContext {
    extensions: AssignedAddOnExtensionListInstance;
    /**
     * Remove a AssignedAddOnInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AssignedAddOnInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssignedAddOnInstance
     */
    fetch(callback?: (error: Error | null, item?: AssignedAddOnInstance) => any): Promise<AssignedAddOnInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AssignedAddOnContextSolution {
    accountSid: string;
    resourceSid: string;
    sid: string;
}
export declare class AssignedAddOnContextImpl implements AssignedAddOnContext {
    protected _version: V2010;
    protected _solution: AssignedAddOnContextSolution;
    protected _uri: string;
    protected _extensions?: AssignedAddOnExtensionListInstance;
    constructor(_version: V2010, accountSid: string, resourceSid: string, sid: string);
    get extensions(): AssignedAddOnExtensionListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AssignedAddOnInstance) => any): Promise<AssignedAddOnInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AssignedAddOnContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AssignedAddOnPayload extends TwilioResponsePayload {
    assigned_add_ons: AssignedAddOnResource[];
}
interface AssignedAddOnResource {
    sid: string;
    account_sid: string;
    resource_sid: string;
    friendly_name: string;
    description: string;
    configuration: any;
    unique_name: string;
    date_created: Date;
    date_updated: Date;
    uri: string;
    subresource_uris: Record<string, string>;
}
export declare class AssignedAddOnInstance {
    protected _version: V2010;
    protected _solution: AssignedAddOnContextSolution;
    protected _context?: AssignedAddOnContext;
    constructor(_version: V2010, payload: AssignedAddOnResource, accountSid: string, resourceSid: string, sid?: string);
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
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * A short description of the functionality that the Add-on provides.
     */
    description: string;
    /**
     * A JSON string that represents the current configuration of this Add-on installation.
     */
    configuration: any;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used in place of the resource\'s `sid` in the URL to address the resource.
     */
    uniqueName: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    /**
     * A list of related resources identified by their relative URIs.
     */
    subresourceUris: Record<string, string>;
    private get _proxy();
    /**
     * Remove a AssignedAddOnInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AssignedAddOnInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssignedAddOnInstance
     */
    fetch(callback?: (error: Error | null, item?: AssignedAddOnInstance) => any): Promise<AssignedAddOnInstance>;
    /**
     * Access the extensions.
     */
    extensions(): AssignedAddOnExtensionListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        resourceSid: string;
        friendlyName: string;
        description: string;
        configuration: any;
        uniqueName: string;
        dateCreated: Date;
        dateUpdated: Date;
        uri: string;
        subresourceUris: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AssignedAddOnSolution {
    accountSid: string;
    resourceSid: string;
}
export interface AssignedAddOnListInstance {
    _version: V2010;
    _solution: AssignedAddOnSolution;
    _uri: string;
    (sid: string): AssignedAddOnContext;
    get(sid: string): AssignedAddOnContext;
    /**
     * Create a AssignedAddOnInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssignedAddOnInstance
     */
    create(params: AssignedAddOnListInstanceCreateOptions, callback?: (error: Error | null, item?: AssignedAddOnInstance) => any): Promise<AssignedAddOnInstance>;
    /**
     * Streams AssignedAddOnInstance records from the API.
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
     * @param { AssignedAddOnListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AssignedAddOnInstance, done: (err?: Error) => void) => void): void;
    each(params: AssignedAddOnListInstanceEachOptions, callback?: (item: AssignedAddOnInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AssignedAddOnInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AssignedAddOnPage) => any): Promise<AssignedAddOnPage>;
    /**
     * Lists AssignedAddOnInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AssignedAddOnListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AssignedAddOnInstance[]) => any): Promise<AssignedAddOnInstance[]>;
    list(params: AssignedAddOnListInstanceOptions, callback?: (error: Error | null, items: AssignedAddOnInstance[]) => any): Promise<AssignedAddOnInstance[]>;
    /**
     * Retrieve a single page of AssignedAddOnInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AssignedAddOnListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AssignedAddOnPage) => any): Promise<AssignedAddOnPage>;
    page(params: AssignedAddOnListInstancePageOptions, callback?: (error: Error | null, items: AssignedAddOnPage) => any): Promise<AssignedAddOnPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AssignedAddOnListInstance(version: V2010, accountSid: string, resourceSid: string): AssignedAddOnListInstance;
export declare class AssignedAddOnPage extends Page<V2010, AssignedAddOnPayload, AssignedAddOnResource, AssignedAddOnInstance> {
    /**
     * Initialize the AssignedAddOnPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: AssignedAddOnSolution);
    /**
     * Build an instance of AssignedAddOnInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AssignedAddOnResource): AssignedAddOnInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
