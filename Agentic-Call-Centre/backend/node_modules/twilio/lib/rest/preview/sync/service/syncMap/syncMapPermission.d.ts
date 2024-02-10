/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import Sync from "../../../Sync";
/**
 * Options to pass to update a SyncMapPermissionInstance
 */
export interface SyncMapPermissionContextUpdateOptions {
    /** Boolean flag specifying whether the identity can read the Sync Map. */
    read: boolean;
    /** Boolean flag specifying whether the identity can create, update and delete Items of the Sync Map. */
    write: boolean;
    /** Boolean flag specifying whether the identity can delete the Sync Map. */
    manage: boolean;
}
/**
 * Options to pass to each
 */
export interface SyncMapPermissionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SyncMapPermissionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SyncMapPermissionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SyncMapPermissionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SyncMapPermissionContext {
    /**
     * Remove a SyncMapPermissionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SyncMapPermissionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapPermissionInstance
     */
    fetch(callback?: (error: Error | null, item?: SyncMapPermissionInstance) => any): Promise<SyncMapPermissionInstance>;
    /**
     * Update a SyncMapPermissionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapPermissionInstance
     */
    update(params: SyncMapPermissionContextUpdateOptions, callback?: (error: Error | null, item?: SyncMapPermissionInstance) => any): Promise<SyncMapPermissionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SyncMapPermissionContextSolution {
    serviceSid: string;
    mapSid: string;
    identity: string;
}
export declare class SyncMapPermissionContextImpl implements SyncMapPermissionContext {
    protected _version: Sync;
    protected _solution: SyncMapPermissionContextSolution;
    protected _uri: string;
    constructor(_version: Sync, serviceSid: string, mapSid: string, identity: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SyncMapPermissionInstance) => any): Promise<SyncMapPermissionInstance>;
    update(params: SyncMapPermissionContextUpdateOptions, callback?: (error: Error | null, item?: SyncMapPermissionInstance) => any): Promise<SyncMapPermissionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SyncMapPermissionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SyncMapPermissionPayload extends TwilioResponsePayload {
    permissions: SyncMapPermissionResource[];
}
interface SyncMapPermissionResource {
    account_sid: string;
    service_sid: string;
    map_sid: string;
    identity: string;
    read: boolean;
    write: boolean;
    manage: boolean;
    url: string;
}
export declare class SyncMapPermissionInstance {
    protected _version: Sync;
    protected _solution: SyncMapPermissionContextSolution;
    protected _context?: SyncMapPermissionContext;
    constructor(_version: Sync, payload: SyncMapPermissionResource, serviceSid: string, mapSid: string, identity?: string);
    /**
     * The unique SID identifier of the Twilio Account.
     */
    accountSid: string;
    /**
     * The unique SID identifier of the Sync Service Instance.
     */
    serviceSid: string;
    /**
     * The unique SID identifier of the Sync Map to which the Permission applies.
     */
    mapSid: string;
    /**
     * Arbitrary string identifier representing a human user associated with an FPA token, assigned by the developer.
     */
    identity: string;
    /**
     * Boolean flag specifying whether the identity can read the Sync Map and its Items.
     */
    read: boolean;
    /**
     * Boolean flag specifying whether the identity can create, update and delete Items of the Sync Map.
     */
    write: boolean;
    /**
     * Boolean flag specifying whether the identity can delete the Sync Map.
     */
    manage: boolean;
    /**
     * Contains an absolute URL for this Sync Map Permission.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a SyncMapPermissionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SyncMapPermissionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapPermissionInstance
     */
    fetch(callback?: (error: Error | null, item?: SyncMapPermissionInstance) => any): Promise<SyncMapPermissionInstance>;
    /**
     * Update a SyncMapPermissionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SyncMapPermissionInstance
     */
    update(params: SyncMapPermissionContextUpdateOptions, callback?: (error: Error | null, item?: SyncMapPermissionInstance) => any): Promise<SyncMapPermissionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        serviceSid: string;
        mapSid: string;
        identity: string;
        read: boolean;
        write: boolean;
        manage: boolean;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SyncMapPermissionSolution {
    serviceSid: string;
    mapSid: string;
}
export interface SyncMapPermissionListInstance {
    _version: Sync;
    _solution: SyncMapPermissionSolution;
    _uri: string;
    (identity: string): SyncMapPermissionContext;
    get(identity: string): SyncMapPermissionContext;
    /**
     * Streams SyncMapPermissionInstance records from the API.
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
     * @param { SyncMapPermissionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SyncMapPermissionInstance, done: (err?: Error) => void) => void): void;
    each(params: SyncMapPermissionListInstanceEachOptions, callback?: (item: SyncMapPermissionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SyncMapPermissionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SyncMapPermissionPage) => any): Promise<SyncMapPermissionPage>;
    /**
     * Lists SyncMapPermissionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SyncMapPermissionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SyncMapPermissionInstance[]) => any): Promise<SyncMapPermissionInstance[]>;
    list(params: SyncMapPermissionListInstanceOptions, callback?: (error: Error | null, items: SyncMapPermissionInstance[]) => any): Promise<SyncMapPermissionInstance[]>;
    /**
     * Retrieve a single page of SyncMapPermissionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SyncMapPermissionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SyncMapPermissionPage) => any): Promise<SyncMapPermissionPage>;
    page(params: SyncMapPermissionListInstancePageOptions, callback?: (error: Error | null, items: SyncMapPermissionPage) => any): Promise<SyncMapPermissionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SyncMapPermissionListInstance(version: Sync, serviceSid: string, mapSid: string): SyncMapPermissionListInstance;
export declare class SyncMapPermissionPage extends Page<Sync, SyncMapPermissionPayload, SyncMapPermissionResource, SyncMapPermissionInstance> {
    /**
     * Initialize the SyncMapPermissionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: Sync, response: Response<string>, solution: SyncMapPermissionSolution);
    /**
     * Build an instance of SyncMapPermissionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SyncMapPermissionResource): SyncMapPermissionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
