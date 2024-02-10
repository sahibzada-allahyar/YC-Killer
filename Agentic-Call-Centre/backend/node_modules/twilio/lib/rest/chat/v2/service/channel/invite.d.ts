/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2 from "../../../V2";
/**
 * Options to pass to create a InviteInstance
 */
export interface InviteListInstanceCreateOptions {
    /** The `identity` value that uniquely identifies the new resource\\\'s [User](https://www.twilio.com/docs/chat/rest/user-resource) within the [Service](https://www.twilio.com/docs/chat/rest/service-resource). See [access tokens](https://www.twilio.com/docs/chat/create-tokens) for more info. */
    identity: string;
    /** The SID of the [Role](https://www.twilio.com/docs/chat/rest/role-resource) assigned to the new member. */
    roleSid?: string;
}
/**
 * Options to pass to each
 */
export interface InviteListInstanceEachOptions {
    /** The [User](https://www.twilio.com/docs/chat/rest/user-resource)\'s `identity` value of the resources to read. See [access tokens](https://www.twilio.com/docs/chat/create-tokens) for more details. */
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
    /** The [User](https://www.twilio.com/docs/chat/rest/user-resource)\'s `identity` value of the resources to read. See [access tokens](https://www.twilio.com/docs/chat/create-tokens) for more details. */
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
    /** The [User](https://www.twilio.com/docs/chat/rest/user-resource)\'s `identity` value of the resources to read. See [access tokens](https://www.twilio.com/docs/chat/create-tokens) for more details. */
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
    protected _version: V2;
    protected _solution: InviteContextSolution;
    protected _uri: string;
    constructor(_version: V2, serviceSid: string, channelSid: string, sid: string);
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
    protected _version: V2;
    protected _solution: InviteContextSolution;
    protected _context?: InviteContext;
    constructor(_version: V2, payload: InviteResource, serviceSid: string, channelSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Invite resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Invite resource.
     */
    accountSid: string;
    /**
     * The SID of the [Channel](https://www.twilio.com/docs/chat/channels) the Invite resource belongs to.
     */
    channelSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/chat/rest/service-resource) the Invite resource is associated with.
     */
    serviceSid: string;
    /**
     * The application-defined string that uniquely identifies the resource\'s [User](https://www.twilio.com/docs/chat/rest/user-resource) within the [Service](https://www.twilio.com/docs/chat/rest/service-resource). See [access tokens](https://www.twilio.com/docs/chat/create-tokens) for more info.
     */
    identity: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The SID of the [Role](https://www.twilio.com/docs/chat/rest/role-resource) assigned to the resource.
     */
    roleSid: string;
    /**
     * The `identity` of the User that created the invite.
     */
    createdBy: string;
    /**
     * The absolute URL of the Invite resource.
     */
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
    _version: V2;
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
export declare function InviteListInstance(version: V2, serviceSid: string, channelSid: string): InviteListInstance;
export declare class InvitePage extends Page<V2, InvitePayload, InviteResource, InviteInstance> {
    /**
     * Initialize the InvitePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: InviteSolution);
    /**
     * Build an instance of InviteInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: InviteResource): InviteInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
