/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import Sync from "../../Sync";
import { DocumentPermissionListInstance } from "./document/documentPermission";
/**
 * Options to pass to update a DocumentInstance
 */
export interface DocumentContextUpdateOptions {
    /**  */
    data: any;
    /** The If-Match HTTP request header */
    ifMatch?: string;
}
/**
 * Options to pass to create a DocumentInstance
 */
export interface DocumentListInstanceCreateOptions {
    /**  */
    uniqueName?: string;
    /**  */
    data?: any;
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
    protected _version: Sync;
    protected _solution: DocumentContextSolution;
    protected _uri: string;
    protected _documentPermissions?: DocumentPermissionListInstance;
    constructor(_version: Sync, serviceSid: string, sid: string);
    get documentPermissions(): DocumentPermissionListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: DocumentInstance) => any): Promise<DocumentInstance>;
    update(params: DocumentContextUpdateOptions, callback?: (error: Error | null, item?: DocumentInstance) => any): Promise<DocumentInstance>;
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
    date_created: Date;
    date_updated: Date;
    created_by: string;
}
export declare class DocumentInstance {
    protected _version: Sync;
    protected _solution: DocumentContextSolution;
    protected _context?: DocumentContext;
    constructor(_version: Sync, payload: DocumentResource, serviceSid: string, sid?: string);
    sid: string;
    uniqueName: string;
    accountSid: string;
    serviceSid: string;
    url: string;
    links: Record<string, string>;
    revision: string;
    data: any;
    dateCreated: Date;
    dateUpdated: Date;
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
    _version: Sync;
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
export declare function DocumentListInstance(version: Sync, serviceSid: string): DocumentListInstance;
export declare class DocumentPage extends Page<Sync, DocumentPayload, DocumentResource, DocumentInstance> {
    /**
     * Initialize the DocumentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: Sync, response: Response<string>, solution: DocumentSolution);
    /**
     * Build an instance of DocumentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DocumentResource): DocumentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
