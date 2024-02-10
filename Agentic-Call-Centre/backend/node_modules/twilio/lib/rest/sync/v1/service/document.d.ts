/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { DocumentPermissionListInstance } from "./document/documentPermission";
/**
 * Options to pass to update a DocumentInstance
 */
export interface DocumentContextUpdateOptions {
    /** The If-Match HTTP request header */
    ifMatch?: string;
    /** A JSON string that represents an arbitrary, schema-less object that the Sync Document stores. Can be up to 16 KiB in length. */
    data?: any;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the Sync Document expires and is deleted (time-to-live). */
    ttl?: number;
}
/**
 * Options to pass to create a DocumentInstance
 */
export interface DocumentListInstanceCreateOptions {
    /** An application-defined string that uniquely identifies the Sync Document */
    uniqueName?: string;
    /** A JSON string that represents an arbitrary, schema-less object that the Sync Document stores. Can be up to 16 KiB in length. */
    data?: any;
    /** How long, [in seconds](https://www.twilio.com/docs/sync/limits#sync-payload-limits), before the Sync Document expires and is deleted (the Sync Document\\\'s time-to-live). */
    ttl?: number;
}
/**
 * Options to pass to each
 */
export interface DocumentListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: DocumentInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface DocumentListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface DocumentListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface DocumentContext {
    documentPermissions: DocumentPermissionListInstance;
    /**
     * Remove a DocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a DocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DocumentInstance
     */
    fetch(callback?: (error: Error | null, item?: DocumentInstance) => any): Promise<DocumentInstance>;
    /**
     * Update a DocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DocumentInstance
     */
    update(callback?: (error: Error | null, item?: DocumentInstance) => any): Promise<DocumentInstance>;
    /**
     * Update a DocumentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DocumentInstance
     */
    update(params: DocumentContextUpdateOptions, callback?: (error: Error | null, item?: DocumentInstance) => any): Promise<DocumentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface DocumentContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class DocumentContextImpl implements DocumentContext {
    protected _version: V1;
    protected _solution: DocumentContextSolution;
    protected _uri: string;
    protected _documentPermissions?: DocumentPermissionListInstance;
    constructor(_version: V1, serviceSid: string, sid: string);
    get documentPermissions(): DocumentPermissionListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: DocumentInstance) => any): Promise<DocumentInstance>;
    update(params?: DocumentContextUpdateOptions | ((error: Error | null, item?: DocumentInstance) => any), callback?: (error: Error | null, item?: DocumentInstance) => any): Promise<DocumentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): DocumentContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface DocumentPayload extends TwilioResponsePayload {
    documents: DocumentResource[];
}
interface DocumentResource {
    sid: string;
    unique_name: string;
    account_sid: string;
    service_sid: string;
    url: string;
    links: Record<string, string>;
    revision: string;
    data: any;
    date_expires: Date;
    date_created: Date;
    date_updated: Date;
    created_by: string;
}
export declare class DocumentInstance {
    protected _version: V1;
    protected _solution: DocumentContextSolution;
    protected _context?: DocumentContext;
    constructor(_version: V1, payload: DocumentResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Document resource.
     */
    sid: string;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used in place of the resource\'s `sid` in the URL to address the resource and can be up to 320 characters long.
     */
    uniqueName: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Document resource.
     */
    accountSid: string;
    /**
     * The SID of the [Sync Service](https://www.twilio.com/docs/sync/api/service) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The absolute URL of the Document resource.
     */
    url: string;
    /**
     * The URLs of resources related to the Sync Document.
     */
    links: Record<string, string>;
    /**
     * The current revision of the Sync Document, represented as a string. The `revision` property is used with conditional updates to ensure data consistency.
     */
    revision: string;
    /**
     * An arbitrary, schema-less object that the Sync Document stores. Can be up to 16 KiB in length.
     */
    data: any;
    /**
     * The date and time in GMT when the Sync Document expires and will be deleted, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. If the Sync Document does not expire, this value is `null`. The Document resource might not be deleted immediately after it expires.
     */
    dateExpires: Date;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The identity of the Sync Document\'s creator. If the Sync Document is created from the client SDK, the value matches the Access Token\'s `identity` field. If the Sync Document was created from the REST API, the value is `system`.
     */
    createdBy: string;
    private get _proxy();
    /**
     * Remove a DocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a DocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DocumentInstance
     */
    fetch(callback?: (error: Error | null, item?: DocumentInstance) => any): Promise<DocumentInstance>;
    /**
     * Update a DocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DocumentInstance
     */
    update(callback?: (error: Error | null, item?: DocumentInstance) => any): Promise<DocumentInstance>;
    /**
     * Update a DocumentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DocumentInstance
     */
    update(params: DocumentContextUpdateOptions, callback?: (error: Error | null, item?: DocumentInstance) => any): Promise<DocumentInstance>;
    /**
     * Access the documentPermissions.
     */
    documentPermissions(): DocumentPermissionListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        uniqueName: string;
        accountSid: string;
        serviceSid: string;
        url: string;
        links: Record<string, string>;
        revision: string;
        data: any;
        dateExpires: Date;
        dateCreated: Date;
        dateUpdated: Date;
        createdBy: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DocumentSolution {
    serviceSid: string;
}
export interface DocumentListInstance {
    _version: V1;
    _solution: DocumentSolution;
    _uri: string;
    (sid: string): DocumentContext;
    get(sid: string): DocumentContext;
    /**
     * Create a DocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DocumentInstance
     */
    create(callback?: (error: Error | null, item?: DocumentInstance) => any): Promise<DocumentInstance>;
    /**
     * Create a DocumentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DocumentInstance
     */
    create(params: DocumentListInstanceCreateOptions, callback?: (error: Error | null, item?: DocumentInstance) => any): Promise<DocumentInstance>;
    /**
     * Streams DocumentInstance records from the API.
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
     * @param { DocumentListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: DocumentInstance, done: (err?: Error) => void) => void): void;
    each(params: DocumentListInstanceEachOptions, callback?: (item: DocumentInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of DocumentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: DocumentPage) => any): Promise<DocumentPage>;
    /**
     * Lists DocumentInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DocumentListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: DocumentInstance[]) => any): Promise<DocumentInstance[]>;
    list(params: DocumentListInstanceOptions, callback?: (error: Error | null, items: DocumentInstance[]) => any): Promise<DocumentInstance[]>;
    /**
     * Retrieve a single page of DocumentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DocumentListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: DocumentPage) => any): Promise<DocumentPage>;
    page(params: DocumentListInstancePageOptions, callback?: (error: Error | null, items: DocumentPage) => any): Promise<DocumentPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DocumentListInstance(version: V1, serviceSid: string): DocumentListInstance;
export declare class DocumentPage extends Page<V1, DocumentPayload, DocumentResource, DocumentInstance> {
    /**
     * Initialize the DocumentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: DocumentSolution);
    /**
     * Build an instance of DocumentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DocumentResource): DocumentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
