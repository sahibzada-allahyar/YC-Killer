/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { BindingListInstance } from "./service/binding";
import { ConfigurationListInstance } from "./service/configuration";
import { ConversationListInstance } from "./service/conversation";
import { ParticipantConversationListInstance } from "./service/participantConversation";
import { RoleListInstance } from "./service/role";
import { UserListInstance } from "./service/user";
/**
 * Options to pass to create a ServiceInstance
 */
export interface ServiceListInstanceCreateOptions {
    /** The human-readable name of this service, limited to 256 characters. Optional. */
    friendlyName: string;
}
/**
 * Options to pass to each
 */
export interface ServiceListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ServiceInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ServiceListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ServiceListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ServiceContext {
    bindings: BindingListInstance;
    configuration: ConfigurationListInstance;
    conversations: ConversationListInstance;
    participantConversations: ParticipantConversationListInstance;
    roles: RoleListInstance;
    users: UserListInstance;
    /**
     * Remove a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    fetch(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ServiceContextSolution {
    sid: string;
}
export declare class ServiceContextImpl implements ServiceContext {
    protected _version: V1;
    protected _solution: ServiceContextSolution;
    protected _uri: string;
    protected _bindings?: BindingListInstance;
    protected _configuration?: ConfigurationListInstance;
    protected _conversations?: ConversationListInstance;
    protected _participantConversations?: ParticipantConversationListInstance;
    protected _roles?: RoleListInstance;
    protected _users?: UserListInstance;
    constructor(_version: V1, sid: string);
    get bindings(): BindingListInstance;
    get configuration(): ConfigurationListInstance;
    get conversations(): ConversationListInstance;
    get participantConversations(): ParticipantConversationListInstance;
    get roles(): RoleListInstance;
    get users(): UserListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ServiceContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ServicePayload extends TwilioResponsePayload {
    services: ServiceResource[];
}
interface ServiceResource {
    account_sid: string;
    sid: string;
    friendly_name: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class ServiceInstance {
    protected _version: V1;
    protected _solution: ServiceContextSolution;
    protected _context?: ServiceContext;
    constructor(_version: V1, payload: ServiceResource, sid?: string);
    /**
     * The unique ID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this service.
     */
    accountSid: string;
    /**
     * A 34 character string that uniquely identifies this resource.
     */
    sid: string;
    /**
     * The human-readable name of this service, limited to 256 characters. Optional.
     */
    friendlyName: string;
    /**
     * The date that this resource was created.
     */
    dateCreated: Date;
    /**
     * The date that this resource was last updated.
     */
    dateUpdated: Date;
    /**
     * An absolute API resource URL for this service.
     */
    url: string;
    /**
     * Contains absolute API resource URLs to access conversations, users, roles, bindings and configuration of this service.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ServiceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    fetch(callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Access the bindings.
     */
    bindings(): BindingListInstance;
    /**
     * Access the configuration.
     */
    configuration(): ConfigurationListInstance;
    /**
     * Access the conversations.
     */
    conversations(): ConversationListInstance;
    /**
     * Access the participantConversations.
     */
    participantConversations(): ParticipantConversationListInstance;
    /**
     * Access the roles.
     */
    roles(): RoleListInstance;
    /**
     * Access the users.
     */
    users(): UserListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        sid: string;
        friendlyName: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ServiceSolution {
}
export interface ServiceListInstance {
    _version: V1;
    _solution: ServiceSolution;
    _uri: string;
    (sid: string): ServiceContext;
    get(sid: string): ServiceContext;
    /**
     * Create a ServiceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ServiceInstance
     */
    create(params: ServiceListInstanceCreateOptions, callback?: (error: Error | null, item?: ServiceInstance) => any): Promise<ServiceInstance>;
    /**
     * Streams ServiceInstance records from the API.
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
     * @param { ServiceListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ServiceInstance, done: (err?: Error) => void) => void): void;
    each(params: ServiceListInstanceEachOptions, callback?: (item: ServiceInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ServiceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ServicePage) => any): Promise<ServicePage>;
    /**
     * Lists ServiceInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ServiceListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ServiceInstance[]) => any): Promise<ServiceInstance[]>;
    list(params: ServiceListInstanceOptions, callback?: (error: Error | null, items: ServiceInstance[]) => any): Promise<ServiceInstance[]>;
    /**
     * Retrieve a single page of ServiceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ServiceListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ServicePage) => any): Promise<ServicePage>;
    page(params: ServiceListInstancePageOptions, callback?: (error: Error | null, items: ServicePage) => any): Promise<ServicePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ServiceListInstance(version: V1): ServiceListInstance;
export declare class ServicePage extends Page<V1, ServicePayload, ServiceResource, ServiceInstance> {
    /**
     * Initialize the ServicePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ServiceSolution);
    /**
     * Build an instance of ServiceInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ServiceResource): ServiceInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
