/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type RoleRoleType = "conversation" | "service";
/**
 * Options to pass to update a RoleInstance
 */
export interface RoleContextUpdateOptions {
    /** A permission that you grant to the role. Only one permission can be granted per parameter. To assign more than one permission, repeat this parameter for each permission value. Note that the update action replaces all previously assigned permissions with those defined in the update action. To remove a permission, do not include it in the subsequent update action. The values for this parameter depend on the role\\\'s `type`. */
    permission: Array<string>;
}
/**
 * Options to pass to create a RoleInstance
 */
export interface RoleListInstanceCreateOptions {
    /** A descriptive string that you create to describe the new resource. It can be up to 64 characters long. */
    friendlyName: string;
    /**  */
    type: RoleRoleType;
    /** A permission that you grant to the new role. Only one permission can be granted per parameter. To assign more than one permission, repeat this parameter for each permission value. The values for this parameter depend on the role\\\'s `type`. */
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
    sid: string;
}
export declare class RoleContextImpl implements RoleContext {
    protected _version: V1;
    protected _solution: RoleContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
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
    chat_service_sid: string;
    friendly_name: string;
    type: RoleRoleType;
    permissions: Array<string>;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class RoleInstance {
    protected _version: V1;
    protected _solution: RoleContextSolution;
    protected _context?: RoleContext;
    constructor(_version: V1, payload: RoleResource, sid?: string);
    /**
     * The unique string that we created to identify the Role resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Role resource.
     */
    accountSid: string;
    /**
     * The SID of the [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource) the Role resource is associated with.
     */
    chatServiceSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    type: RoleRoleType;
    /**
     * An array of the permissions the role has been granted.
     */
    permissions: Array<string>;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * An absolute API resource URL for this user role.
     */
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
        chatServiceSid: string;
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
}
export interface RoleListInstance {
    _version: V1;
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
export declare function RoleListInstance(version: V1): RoleListInstance;
export declare class RolePage extends Page<V1, RolePayload, RoleResource, RoleInstance> {
    /**
     * Initialize the RolePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: RoleSolution);
    /**
     * Build an instance of RoleInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: RoleResource): RoleInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
