/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
/**
 * Options to pass to create a InviteInstance
 */
export interface InviteListInstanceCreateOptions {
    /**  */
    identity: string;
    /**  */
    roleSid?: string;
}
/**
 * Options to pass to each
 */
export interface InviteListInstanceEachOptions {
    /**  */
    identity?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: InviteInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface InviteListInstanceOptions {
    /**  */
    identity?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface InviteListInstancePageOptions {
    /**  */
    identity?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface InviteContext {
    /**
     * Remove a InviteInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a InviteInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InviteInstance
     */
    fetch(callback?: (error: Error | null, item?: InviteInstance) => any): Promise<InviteInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface InviteContextSolution {
    serviceSid: string;
    channelSid: string;
    sid: string;
}
export declare class InviteContextImpl implements InviteContext {
    protected _version: V1;
    protected _solution: InviteContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, channelSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: InviteInstance) => any): Promise<InviteInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): InviteContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface InvitePayload extends TwilioResponsePayload {
    invites: InviteResource[];
}
interface InviteResource {
    sid: string;
    account_sid: string;
    channel_sid: string;
    service_sid: string;
    identity: string;
    date_created: Date;
    date_updated: Date;
    role_sid: string;
    created_by: string;
    url: string;
}
export declare class InviteInstance {
    protected _version: V1;
    protected _solution: InviteContextSolution;
    protected _context?: InviteContext;
    constructor(_version: V1, payload: InviteResource, serviceSid: string, channelSid: string, sid?: string);
    sid: string;
    accountSid: string;
    channelSid: string;
    serviceSid: string;
    identity: string;
    dateCreated: Date;
    dateUpdated: Date;
    roleSid: string;
    createdBy: string;
    url: string;
    private get _proxy();
    /**
     * Remove a InviteInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a InviteInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InviteInstance
     */
    fetch(callback?: (error: Error | null, item?: InviteInstance) => any): Promise<InviteInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        channelSid: string;
        serviceSid: string;
        identity: string;
        dateCreated: Date;
        dateUpdated: Date;
        roleSid: string;
        createdBy: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface InviteSolution {
    serviceSid: string;
    channelSid: string;
}
export interface InviteListInstance {
    _version: V1;
    _solution: InviteSolution;
    _uri: string;
    (sid: string): InviteContext;
    get(sid: string): InviteContext;
    /**
     * Create a InviteInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InviteInstance
     */
    create(params: InviteListInstanceCreateOptions, callback?: (error: Error | null, item?: InviteInstance) => any): Promise<InviteInstance>;
    /**
     * Streams InviteInstance records from the API.
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
     * @param { InviteListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: InviteInstance, done: (err?: Error) => void) => void): void;
    each(params: InviteListInstanceEachOptions, callback?: (item: InviteInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of InviteInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: InvitePage) => any): Promise<InvitePage>;
    /**
     * Lists InviteInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InviteListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: InviteInstance[]) => any): Promise<InviteInstance[]>;
    list(params: InviteListInstanceOptions, callback?: (error: Error | null, items: InviteInstance[]) => any): Promise<InviteInstance[]>;
    /**
     * Retrieve a single page of InviteInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InviteListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: InvitePage) => any): Promise<InvitePage>;
    page(params: InviteListInstancePageOptions, callback?: (error: Error | null, items: InvitePage) => any): Promise<InvitePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InviteListInstance(version: V1, serviceSid: string, channelSid: string): InviteListInstance;
export declare class InvitePage extends Page<V1, InvitePayload, InviteResource, InviteInstance> {
    /**
     * Initialize the InvitePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: InviteSolution);
    /**
     * Build an instance of InviteInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: InviteResource): InviteInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
