/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type SupportingDocumentStatus = "draft" | "pending-review" | "rejected" | "approved" | "expired" | "provisionally-approved";
/**
 * Options to pass to update a SupportingDocumentInstance
 */
export interface SupportingDocumentContextUpdateOptions {
    /** The string that you assigned to describe the resource. */
    friendlyName?: string;
    /** The set of parameters that are the attributes of the Supporting Document resource which are derived Supporting Document Types. */
    attributes?: any;
}
/**
 * Options to pass to create a SupportingDocumentInstance
 */
export interface SupportingDocumentListInstanceCreateOptions {
    /** The string that you assigned to describe the resource. */
    friendlyName: string;
    /** The type of the Supporting Document. */
    type: string;
    /** The set of parameters that are the attributes of the Supporting Documents resource which are derived Supporting Document Types. */
    attributes?: any;
}
/**
 * Options to pass to each
 */
export interface SupportingDocumentListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SupportingDocumentInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SupportingDocumentListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SupportingDocumentListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SupportingDocumentContext {
    /**
     * Remove a SupportingDocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SupportingDocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SupportingDocumentInstance
     */
    fetch(callback?: (error: Error | null, item?: SupportingDocumentInstance) => any): Promise<SupportingDocumentInstance>;
    /**
     * Update a SupportingDocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SupportingDocumentInstance
     */
    update(callback?: (error: Error | null, item?: SupportingDocumentInstance) => any): Promise<SupportingDocumentInstance>;
    /**
     * Update a SupportingDocumentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SupportingDocumentInstance
     */
    update(params: SupportingDocumentContextUpdateOptions, callback?: (error: Error | null, item?: SupportingDocumentInstance) => any): Promise<SupportingDocumentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SupportingDocumentContextSolution {
    sid: string;
}
export declare class SupportingDocumentContextImpl implements SupportingDocumentContext {
    protected _version: V1;
    protected _solution: SupportingDocumentContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SupportingDocumentInstance) => any): Promise<SupportingDocumentInstance>;
    update(params?: SupportingDocumentContextUpdateOptions | ((error: Error | null, item?: SupportingDocumentInstance) => any), callback?: (error: Error | null, item?: SupportingDocumentInstance) => any): Promise<SupportingDocumentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SupportingDocumentContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SupportingDocumentPayload extends TwilioResponsePayload {
    results: SupportingDocumentResource[];
}
interface SupportingDocumentResource {
    sid: string;
    account_sid: string;
    friendly_name: string;
    mime_type: string;
    status: SupportingDocumentStatus;
    type: string;
    attributes: any;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class SupportingDocumentInstance {
    protected _version: V1;
    protected _solution: SupportingDocumentContextSolution;
    protected _context?: SupportingDocumentContext;
    constructor(_version: V1, payload: SupportingDocumentResource, sid?: string);
    /**
     * The unique string created by Twilio to identify the Supporting Document resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Document resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The image type uploaded in the Supporting Document container.
     */
    mimeType: string;
    status: SupportingDocumentStatus;
    /**
     * The type of the Supporting Document.
     */
    type: string;
    /**
     * The set of parameters that are the attributes of the Supporting Documents resource which are listed in the Supporting Document Types.
     */
    attributes: any;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Supporting Document resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a SupportingDocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SupportingDocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SupportingDocumentInstance
     */
    fetch(callback?: (error: Error | null, item?: SupportingDocumentInstance) => any): Promise<SupportingDocumentInstance>;
    /**
     * Update a SupportingDocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SupportingDocumentInstance
     */
    update(callback?: (error: Error | null, item?: SupportingDocumentInstance) => any): Promise<SupportingDocumentInstance>;
    /**
     * Update a SupportingDocumentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SupportingDocumentInstance
     */
    update(params: SupportingDocumentContextUpdateOptions, callback?: (error: Error | null, item?: SupportingDocumentInstance) => any): Promise<SupportingDocumentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        friendlyName: string;
        mimeType: string;
        status: SupportingDocumentStatus;
        type: string;
        attributes: any;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SupportingDocumentSolution {
}
export interface SupportingDocumentListInstance {
    _version: V1;
    _solution: SupportingDocumentSolution;
    _uri: string;
    (sid: string): SupportingDocumentContext;
    get(sid: string): SupportingDocumentContext;
    /**
     * Create a SupportingDocumentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SupportingDocumentInstance
     */
    create(params: SupportingDocumentListInstanceCreateOptions, callback?: (error: Error | null, item?: SupportingDocumentInstance) => any): Promise<SupportingDocumentInstance>;
    /**
     * Streams SupportingDocumentInstance records from the API.
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
     * @param { SupportingDocumentListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SupportingDocumentInstance, done: (err?: Error) => void) => void): void;
    each(params: SupportingDocumentListInstanceEachOptions, callback?: (item: SupportingDocumentInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SupportingDocumentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SupportingDocumentPage) => any): Promise<SupportingDocumentPage>;
    /**
     * Lists SupportingDocumentInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SupportingDocumentListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SupportingDocumentInstance[]) => any): Promise<SupportingDocumentInstance[]>;
    list(params: SupportingDocumentListInstanceOptions, callback?: (error: Error | null, items: SupportingDocumentInstance[]) => any): Promise<SupportingDocumentInstance[]>;
    /**
     * Retrieve a single page of SupportingDocumentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SupportingDocumentListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SupportingDocumentPage) => any): Promise<SupportingDocumentPage>;
    page(params: SupportingDocumentListInstancePageOptions, callback?: (error: Error | null, items: SupportingDocumentPage) => any): Promise<SupportingDocumentPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SupportingDocumentListInstance(version: V1): SupportingDocumentListInstance;
export declare class SupportingDocumentPage extends Page<V1, SupportingDocumentPayload, SupportingDocumentResource, SupportingDocumentInstance> {
    /**
     * Initialize the SupportingDocumentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SupportingDocumentSolution);
    /**
     * Build an instance of SupportingDocumentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SupportingDocumentResource): SupportingDocumentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
