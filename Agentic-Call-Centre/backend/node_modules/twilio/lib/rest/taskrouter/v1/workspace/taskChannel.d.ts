/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to update a TaskChannelInstance
 */
export interface TaskChannelContextUpdateOptions {
    /** A descriptive string that you create to describe the Task Channel. It can be up to 64 characters long. */
    friendlyName?: string;
    /** Whether the TaskChannel should prioritize Workers that have been idle. If `true`, Workers that have been idle the longest are prioritized. */
    channelOptimizedRouting?: boolean;
}
/**
 * Options to pass to create a TaskChannelInstance
 */
export interface TaskChannelListInstanceCreateOptions {
    /** A descriptive string that you create to describe the Task Channel. It can be up to 64 characters long. */
    friendlyName: string;
    /** An application-defined string that uniquely identifies the Task Channel, such as `voice` or `sms`. */
    uniqueName: string;
    /** Whether the Task Channel should prioritize Workers that have been idle. If `true`, Workers that have been idle the longest are prioritized. */
    channelOptimizedRouting?: boolean;
}
/**
 * Options to pass to each
 */
export interface TaskChannelListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TaskChannelInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TaskChannelListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TaskChannelListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TaskChannelContext {
    /**
     * Remove a TaskChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TaskChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: TaskChannelInstance) => any): Promise<TaskChannelInstance>;
    /**
     * Update a TaskChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskChannelInstance
     */
    update(callback?: (error: Error | null, item?: TaskChannelInstance) => any): Promise<TaskChannelInstance>;
    /**
     * Update a TaskChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskChannelInstance
     */
    update(params: TaskChannelContextUpdateOptions, callback?: (error: Error | null, item?: TaskChannelInstance) => any): Promise<TaskChannelInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TaskChannelContextSolution {
    workspaceSid: string;
    sid: string;
}
export declare class TaskChannelContextImpl implements TaskChannelContext {
    protected _version: V1;
    protected _solution: TaskChannelContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: TaskChannelInstance) => any): Promise<TaskChannelInstance>;
    update(params?: TaskChannelContextUpdateOptions | ((error: Error | null, item?: TaskChannelInstance) => any), callback?: (error: Error | null, item?: TaskChannelInstance) => any): Promise<TaskChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TaskChannelContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TaskChannelPayload extends TwilioResponsePayload {
    channels: TaskChannelResource[];
}
interface TaskChannelResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    sid: string;
    unique_name: string;
    workspace_sid: string;
    channel_optimized_routing: boolean;
    url: string;
    links: Record<string, string>;
}
export declare class TaskChannelInstance {
    protected _version: V1;
    protected _solution: TaskChannelContextSolution;
    protected _context?: TaskChannelContext;
    constructor(_version: V1, payload: TaskChannelResource, workspaceSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Task Channel resource.
     */
    accountSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The unique string that we created to identify the Task Channel resource.
     */
    sid: string;
    /**
     * An application-defined string that uniquely identifies the Task Channel, such as `voice` or `sms`.
     */
    uniqueName: string;
    /**
     * The SID of the Workspace that contains the Task Channel.
     */
    workspaceSid: string;
    /**
     * Whether the Task Channel will prioritize Workers that have been idle. When `true`, Workers that have been idle the longest are prioritized.
     */
    channelOptimizedRouting: boolean;
    /**
     * The absolute URL of the Task Channel resource.
     */
    url: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a TaskChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TaskChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: TaskChannelInstance) => any): Promise<TaskChannelInstance>;
    /**
     * Update a TaskChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskChannelInstance
     */
    update(callback?: (error: Error | null, item?: TaskChannelInstance) => any): Promise<TaskChannelInstance>;
    /**
     * Update a TaskChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskChannelInstance
     */
    update(params: TaskChannelContextUpdateOptions, callback?: (error: Error | null, item?: TaskChannelInstance) => any): Promise<TaskChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        sid: string;
        uniqueName: string;
        workspaceSid: string;
        channelOptimizedRouting: boolean;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TaskChannelSolution {
    workspaceSid: string;
}
export interface TaskChannelListInstance {
    _version: V1;
    _solution: TaskChannelSolution;
    _uri: string;
    (sid: string): TaskChannelContext;
    get(sid: string): TaskChannelContext;
    /**
     * Create a TaskChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskChannelInstance
     */
    create(params: TaskChannelListInstanceCreateOptions, callback?: (error: Error | null, item?: TaskChannelInstance) => any): Promise<TaskChannelInstance>;
    /**
     * Streams TaskChannelInstance records from the API.
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
     * @param { TaskChannelListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TaskChannelInstance, done: (err?: Error) => void) => void): void;
    each(params: TaskChannelListInstanceEachOptions, callback?: (item: TaskChannelInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TaskChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TaskChannelPage) => any): Promise<TaskChannelPage>;
    /**
     * Lists TaskChannelInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TaskChannelListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TaskChannelInstance[]) => any): Promise<TaskChannelInstance[]>;
    list(params: TaskChannelListInstanceOptions, callback?: (error: Error | null, items: TaskChannelInstance[]) => any): Promise<TaskChannelInstance[]>;
    /**
     * Retrieve a single page of TaskChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TaskChannelListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TaskChannelPage) => any): Promise<TaskChannelPage>;
    page(params: TaskChannelListInstancePageOptions, callback?: (error: Error | null, items: TaskChannelPage) => any): Promise<TaskChannelPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TaskChannelListInstance(version: V1, workspaceSid: string): TaskChannelListInstance;
export declare class TaskChannelPage extends Page<V1, TaskChannelPayload, TaskChannelResource, TaskChannelInstance> {
    /**
     * Initialize the TaskChannelPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: TaskChannelSolution);
    /**
     * Build an instance of TaskChannelInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TaskChannelResource): TaskChannelInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
