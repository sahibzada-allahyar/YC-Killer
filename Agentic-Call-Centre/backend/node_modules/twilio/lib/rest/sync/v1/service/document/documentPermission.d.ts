/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
/**
 * Options to pass to update a DocumentPermissionInstance
 */
export interface DocumentPermissionContextUpdateOptions {
    /** Whether the identity can read the Sync Document. Default value is `false`. */
    read: boolean;
    /** Whether the identity can update the Sync Document. Default value is `false`. */
    write: boolean;
    /** Whether the identity can delete the Sync Document. Default value is `false`. */
    manage: boolean;
}
/**
 * Options to pass to each
 */
export interface DocumentPermissionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: DocumentPermissionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface DocumentPermissionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface DocumentPermissionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface DocumentPermissionContext {
    /**
     * Remove a DocumentPermissionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a DocumentPermissionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DocumentPermissionInstance
     */
    fetch(callback?: (error: Error | null, item?: DocumentPermissionInstance) => any): Promise<DocumentPermissionInstance>;
    /**
     * Update a DocumentPermissionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DocumentPermissionInstance
     */
    update(params: DocumentPermissionContextUpdateOptions, callback?: (error: Error | null, item?: DocumentPermissionInstance) => any): Promise<DocumentPermissionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface DocumentPermissionContextSolution {
    serviceSid: string;
    documentSid: string;
    identity: string;
}
export declare class DocumentPermissionContextImpl implements DocumentPermissionContext {
    protected _version: V1;
    protected _solution: DocumentPermissionContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, documentSid: string, identity: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: DocumentPermissionInstance) => any): Promise<DocumentPermissionInstance>;
    update(params: DocumentPermissionContextUpdateOptions, callback?: (error: Error | null, item?: DocumentPermissionInstance) => any): Promise<DocumentPermissionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): DocumentPermissionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface DocumentPermissionPayload extends TwilioResponsePayload {
    permissions: DocumentPermissionResource[];
}
interface DocumentPermissionResource {
    account_sid: string;
    service_sid: string;
    document_sid: string;
    identity: string;
    read: boolean;
    write: boolean;
    manage: boolean;
    url: string;
}
export declare class DocumentPermissionInstance {
    protected _version: V1;
    protected _solution: DocumentPermissionContextSolution;
    protected _context?: DocumentPermissionContext;
    constructor(_version: V1, payload: DocumentPermissionResource, serviceSid: string, documentSid: string, identity?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Document Permission resource.
     */
    accountSid: string;
    /**
     * The SID of the [Sync Service](https://www.twilio.com/docs/sync/api/service) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the Sync Document to which the Document Permission applies.
     */
    documentSid: string;
    /**
     * The application-defined string that uniquely identifies the resource\'s User within the Service to an FPA token.
     */
    identity: string;
    /**
     * Whether the identity can read the Sync Document.
     */
    read: boolean;
    /**
     * Whether the identity can update the Sync Document.
     */
    write: boolean;
    /**
     * Whether the identity can delete the Sync Document.
     */
    manage: boolean;
    /**
     * The absolute URL of the Sync Document Permission resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a DocumentPermissionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a DocumentPermissionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DocumentPermissionInstance
     */
    fetch(callback?: (error: Error | null, item?: DocumentPermissionInstance) => any): Promise<DocumentPermissionInstance>;
    /**
     * Update a DocumentPermissionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DocumentPermissionInstance
     */
    update(params: DocumentPermissionContextUpdateOptions, callback?: (error: Error | null, item?: DocumentPermissionInstance) => any): Promise<DocumentPermissionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        serviceSid: string;
        documentSid: string;
        identity: string;
        read: boolean;
        write: boolean;
        manage: boolean;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DocumentPermissionSolution {
    serviceSid: string;
    documentSid: string;
}
export interface DocumentPermissionListInstance {
    _version: V1;
    _solution: DocumentPermissionSolution;
    _uri: string;
    (identity: string): DocumentPermissionContext;
    get(identity: string): DocumentPermissionContext;
    /**
     * Streams DocumentPermissionInstance records from the API.
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
     * @param { DocumentPermissionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: DocumentPermissionInstance, done: (err?: Error) => void) => void): void;
    each(params: DocumentPermissionListInstanceEachOptions, callback?: (item: DocumentPermissionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of DocumentPermissionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: DocumentPermissionPage) => any): Promise<DocumentPermissionPage>;
    /**
     * Lists DocumentPermissionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DocumentPermissionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: DocumentPermissionInstance[]) => any): Promise<DocumentPermissionInstance[]>;
    list(params: DocumentPermissionListInstanceOptions, callback?: (error: Error | null, items: DocumentPermissionInstance[]) => any): Promise<DocumentPermissionInstance[]>;
    /**
     * Retrieve a single page of DocumentPermissionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DocumentPermissionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: DocumentPermissionPage) => any): Promise<DocumentPermissionPage>;
    page(params: DocumentPermissionListInstancePageOptions, callback?: (error: Error | null, items: DocumentPermissionPage) => any): Promise<DocumentPermissionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DocumentPermissionListInstance(version: V1, serviceSid: string, documentSid: string): DocumentPermissionListInstance;
export declare class DocumentPermissionPage extends Page<V1, DocumentPermissionPayload, DocumentPermissionResource, DocumentPermissionInstance> {
    /**
     * Initialize the DocumentPermissionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: DocumentPermissionSolution);
    /**
     * Build an instance of DocumentPermissionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DocumentPermissionResource): DocumentPermissionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
