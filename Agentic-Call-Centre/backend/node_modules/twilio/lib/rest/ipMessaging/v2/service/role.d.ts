/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
export type RoleRoleType = "channel" | "deployment";
/**
 * Options to pass to update a RoleInstance
 */
export interface RoleContextUpdateOptions {
    /**  */
    permission: Array<string>;
}
/**
 * Options to pass to create a RoleInstance
 */
export interface RoleListInstanceCreateOptions {
    /**  */
    friendlyName: string;
    /**  */
    type: RoleRoleType;
    /**  */
    permission: Array<string>;
}
/**
 * Options to pass to each
 */
export interface RoleListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: RoleInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface RoleListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface RoleListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface RoleContext {
    /**
     * Remove a RoleInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RoleInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoleInstance
     */
    fetch(callback?: (error: Error | null, item?: RoleInstance) => any): Promise<RoleInstance>;
    /**
     * Update a RoleInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoleInstance
     */
    update(params: RoleContextUpdateOptions, callback?: (error: Error | null, item?: RoleInstance) => any): Promise<RoleInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface RoleContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class RoleContextImpl implements RoleContext {
    protected _version: V2;
    protected _solution: RoleContextSolution;
    protected _uri: string;
    constructor(_version: V2, serviceSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: RoleInstance) => any): Promise<RoleInstance>;
    update(params: RoleContextUpdateOptions, callback?: (error: Error | null, item?: RoleInstance) => any): Promise<RoleInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): RoleContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface RolePayload extends TwilioResponsePayload {
    roles: RoleResource[];
}
interface RoleResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    friendly_name: string;
    type: RoleRoleType;
    permissions: Array<string>;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class RoleInstance {
    protected _version: V2;
    protected _solution: RoleContextSolution;
    protected _context?: RoleContext;
    constructor(_version: V2, payload: RoleResource, serviceSid: string, sid?: string);
    sid: string;
    accountSid: string;
    serviceSid: string;
    friendlyName: string;
    type: RoleRoleType;
    permissions: Array<string>;
    dateCreated: Date;
    dateUpdated: Date;
    url: string;
    private get _proxy();
    /**
     * Remove a RoleInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RoleInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoleInstance
     */
    fetch(callback?: (error: Error | null, item?: RoleInstance) => any): Promise<RoleInstance>;
    /**
     * Update a RoleInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoleInstance
     */
    update(params: RoleContextUpdateOptions, callback?: (error: Error | null, item?: RoleInstance) => any): Promise<RoleInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        friendlyName: string;
        type: RoleRoleType;
        permissions: string[];
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface RoleSolution {
    serviceSid: string;
}
export interface RoleListInstance {
    _version: V2;
    _solution: RoleSolution;
    _uri: string;
    (sid: string): RoleContext;
    get(sid: string): RoleContext;
    /**
     * Create a RoleInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RoleInstance
     */
    create(params: RoleListInstanceCreateOptions, callback?: (error: Error | null, item?: RoleInstance) => any): Promise<RoleInstance>;
    /**
     * Streams RoleInstance records from the API.
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
     * @param { RoleListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: RoleInstance, done: (err?: Error) => void) => void): void;
    each(params: RoleListInstanceEachOptions, callback?: (item: RoleInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of RoleInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: RolePage) => any): Promise<RolePage>;
    /**
     * Lists RoleInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RoleListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: RoleInstance[]) => any): Promise<RoleInstance[]>;
    list(params: RoleListInstanceOptions, callback?: (error: Error | null, items: RoleInstance[]) => any): Promise<RoleInstance[]>;
    /**
     * Retrieve a single page of RoleInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RoleListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: RolePage) => any): Promise<RolePage>;
    page(params: RoleListInstancePageOptions, callback?: (error: Error | null, items: RolePage) => any): Promise<RolePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RoleListInstance(version: V2, serviceSid: string): RoleListInstance;
export declare class RolePage extends Page<V2, RolePayload, RoleResource, RoleInstance> {
    /**
     * Initialize the RolePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: RoleSolution);
    /**
     * Build an instance of RoleInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: RoleResource): RoleInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
