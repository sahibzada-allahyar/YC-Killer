/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
import { InviteListInstance } from "./channel/invite";
import { MemberListInstance } from "./channel/member";
import { MessageListInstance } from "./channel/message";
import { WebhookListInstance } from "./channel/webhook";
export type ChannelChannelType = "public" | "private";
export type ChannelWebhookEnabledType = "true" | "false";
/**
 * Options to pass to remove a ChannelInstance
 */
export interface ChannelContextRemoveOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: ChannelWebhookEnabledType;
}
/**
 * Options to pass to update a ChannelInstance
 */
export interface ChannelContextUpdateOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: ChannelWebhookEnabledType;
    /**  */
    friendlyName?: string;
    /**  */
    uniqueName?: string;
    /**  */
    attributes?: string;
    /**  */
    dateCreated?: Date;
    /**  */
    dateUpdated?: Date;
    /**  */
    createdBy?: string;
}
/**
 * Options to pass to create a ChannelInstance
 */
export interface ChannelListInstanceCreateOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: ChannelWebhookEnabledType;
    /**  */
    friendlyName?: string;
    /**  */
    uniqueName?: string;
    /**  */
    attributes?: string;
    /**  */
    type?: ChannelChannelType;
    /**  */
    dateCreated?: Date;
    /**  */
    dateUpdated?: Date;
    /**  */
    createdBy?: string;
}
/**
 * Options to pass to each
 */
export interface ChannelListInstanceEachOptions {
    /**  */
    type?: Array<ChannelChannelType>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ChannelInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ChannelListInstanceOptions {
    /**  */
    type?: Array<ChannelChannelType>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ChannelListInstancePageOptions {
    /**  */
    type?: Array<ChannelChannelType>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ChannelContext {
    invites: InviteListInstance;
    members: MemberListInstance;
    messages: MessageListInstance;
    webhooks: WebhookListInstance;
    /**
     * Remove a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a ChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    remove(params: ChannelContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Update a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    update(callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Update a ChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    update(params: ChannelContextUpdateOptions, callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ChannelContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class ChannelContextImpl implements ChannelContext {
    protected _version: V2;
    protected _solution: ChannelContextSolution;
    protected _uri: string;
    protected _invites?: InviteListInstance;
    protected _members?: MemberListInstance;
    protected _messages?: MessageListInstance;
    protected _webhooks?: WebhookListInstance;
    constructor(_version: V2, serviceSid: string, sid: string);
    get invites(): InviteListInstance;
    get members(): MemberListInstance;
    get messages(): MessageListInstance;
    get webhooks(): WebhookListInstance;
    remove(params?: ChannelContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    update(params?: ChannelContextUpdateOptions | ((error: Error | null, item?: ChannelInstance) => any), callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ChannelContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ChannelPayload extends TwilioResponsePayload {
    channels: ChannelResource[];
}
interface ChannelResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    friendly_name: string;
    unique_name: string;
    attributes: string;
    type: ChannelChannelType;
    date_created: Date;
    date_updated: Date;
    created_by: string;
    members_count: number;
    messages_count: number;
    url: string;
    links: Record<string, string>;
}
export declare class ChannelInstance {
    protected _version: V2;
    protected _solution: ChannelContextSolution;
    protected _context?: ChannelContext;
    constructor(_version: V2, payload: ChannelResource, serviceSid: string, sid?: string);
    sid: string;
    accountSid: string;
    serviceSid: string;
    friendlyName: string;
    uniqueName: string;
    attributes: string;
    type: ChannelChannelType;
    dateCreated: Date;
    dateUpdated: Date;
    createdBy: string;
    membersCount: number;
    messagesCount: number;
    url: string;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a ChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    remove(params: ChannelContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Update a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    update(callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Update a ChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    update(params: ChannelContextUpdateOptions, callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Access the invites.
     */
    invites(): InviteListInstance;
    /**
     * Access the members.
     */
    members(): MemberListInstance;
    /**
     * Access the messages.
     */
    messages(): MessageListInstance;
    /**
     * Access the webhooks.
     */
    webhooks(): WebhookListInstance;
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
        uniqueName: string;
        attributes: string;
        type: ChannelChannelType;
        dateCreated: Date;
        dateUpdated: Date;
        createdBy: string;
        membersCount: number;
        messagesCount: number;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ChannelSolution {
    serviceSid: string;
}
export interface ChannelListInstance {
    _version: V2;
    _solution: ChannelSolution;
    _uri: string;
    (sid: string): ChannelContext;
    get(sid: string): ChannelContext;
    /**
     * Create a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    create(callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Create a ChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    create(params: ChannelListInstanceCreateOptions, callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Streams ChannelInstance records from the API.
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
     * @param { ChannelListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ChannelInstance, done: (err?: Error) => void) => void): void;
    each(params: ChannelListInstanceEachOptions, callback?: (item: ChannelInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ChannelPage) => any): Promise<ChannelPage>;
    /**
     * Lists ChannelInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ChannelListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ChannelInstance[]) => any): Promise<ChannelInstance[]>;
    list(params: ChannelListInstanceOptions, callback?: (error: Error | null, items: ChannelInstance[]) => any): Promise<ChannelInstance[]>;
    /**
     * Retrieve a single page of ChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ChannelListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ChannelPage) => any): Promise<ChannelPage>;
    page(params: ChannelListInstancePageOptions, callback?: (error: Error | null, items: ChannelPage) => any): Promise<ChannelPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ChannelListInstance(version: V2, serviceSid: string): ChannelListInstance;
export declare class ChannelPage extends Page<V2, ChannelPayload, ChannelResource, ChannelInstance> {
    /**
     * Initialize the ChannelPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: ChannelSolution);
    /**
     * Build an instance of ChannelInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ChannelResource): ChannelInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
