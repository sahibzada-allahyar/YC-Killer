/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
/**
 * Options to pass to update a WorkerChannelInstance
 */
export interface WorkerChannelContextUpdateOptions {
    /** The total number of Tasks that the Worker should handle for the TaskChannel type. TaskRouter creates reservations for Tasks of this TaskChannel type up to the specified capacity. If the capacity is 0, no new reservations will be created. */
    capacity?: number;
    /** Whether the WorkerChannel is available. Set to `false` to prevent the Worker from receiving any new Tasks of this TaskChannel type. */
    available?: boolean;
}
/**
 * Options to pass to each
 */
export interface WorkerChannelListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: WorkerChannelInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface WorkerChannelListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface WorkerChannelListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface WorkerChannelContext {
    /**
     * Fetch a WorkerChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkerChannelInstance) => any): Promise<WorkerChannelInstance>;
    /**
     * Update a WorkerChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerChannelInstance
     */
    update(callback?: (error: Error | null, item?: WorkerChannelInstance) => any): Promise<WorkerChannelInstance>;
    /**
     * Update a WorkerChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerChannelInstance
     */
    update(params: WorkerChannelContextUpdateOptions, callback?: (error: Error | null, item?: WorkerChannelInstance) => any): Promise<WorkerChannelInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WorkerChannelContextSolution {
    workspaceSid: string;
    workerSid: string;
    sid: string;
}
export declare class WorkerChannelContextImpl implements WorkerChannelContext {
    protected _version: V1;
    protected _solution: WorkerChannelContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string, workerSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: WorkerChannelInstance) => any): Promise<WorkerChannelInstance>;
    update(params?: WorkerChannelContextUpdateOptions | ((error: Error | null, item?: WorkerChannelInstance) => any), callback?: (error: Error | null, item?: WorkerChannelInstance) => any): Promise<WorkerChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WorkerChannelContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WorkerChannelPayload extends TwilioResponsePayload {
    channels: WorkerChannelResource[];
}
interface WorkerChannelResource {
    account_sid: string;
    assigned_tasks: number;
    available: boolean;
    available_capacity_percentage: number;
    configured_capacity: number;
    date_created: Date;
    date_updated: Date;
    sid: string;
    task_channel_sid: string;
    task_channel_unique_name: string;
    worker_sid: string;
    workspace_sid: string;
    url: string;
}
export declare class WorkerChannelInstance {
    protected _version: V1;
    protected _solution: WorkerChannelContextSolution;
    protected _context?: WorkerChannelContext;
    constructor(_version: V1, payload: WorkerChannelResource, workspaceSid: string, workerSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Worker resource.
     */
    accountSid: string;
    /**
     * The total number of Tasks assigned to Worker for the TaskChannel type.
     */
    assignedTasks: number;
    /**
     * Whether the Worker should receive Tasks of the TaskChannel type.
     */
    available: boolean;
    /**
     * The current percentage of capacity the TaskChannel has available. Can be a number between `0` and `100`. A value of `0` indicates that TaskChannel has no capacity available and a value of `100` means the  Worker is available to receive any Tasks of this TaskChannel type.
     */
    availableCapacityPercentage: number;
    /**
     * The current configured capacity for the WorkerChannel. TaskRouter will not create any reservations after the assigned Tasks for the Worker reaches the value.
     */
    configuredCapacity: number;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The unique string that we created to identify the WorkerChannel resource.
     */
    sid: string;
    /**
     * The SID of the TaskChannel.
     */
    taskChannelSid: string;
    /**
     * The unique name of the TaskChannel, such as `voice` or `sms`.
     */
    taskChannelUniqueName: string;
    /**
     * The SID of the Worker that contains the WorkerChannel.
     */
    workerSid: string;
    /**
     * The SID of the Workspace that contains the WorkerChannel.
     */
    workspaceSid: string;
    /**
     * The absolute URL of the WorkerChannel resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a WorkerChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkerChannelInstance) => any): Promise<WorkerChannelInstance>;
    /**
     * Update a WorkerChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerChannelInstance
     */
    update(callback?: (error: Error | null, item?: WorkerChannelInstance) => any): Promise<WorkerChannelInstance>;
    /**
     * Update a WorkerChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerChannelInstance
     */
    update(params: WorkerChannelContextUpdateOptions, callback?: (error: Error | null, item?: WorkerChannelInstance) => any): Promise<WorkerChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        assignedTasks: number;
        available: boolean;
        availableCapacityPercentage: number;
        configuredCapacity: number;
        dateCreated: Date;
        dateUpdated: Date;
        sid: string;
        taskChannelSid: string;
        taskChannelUniqueName: string;
        workerSid: string;
        workspaceSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WorkerChannelSolution {
    workspaceSid: string;
    workerSid: string;
}
export interface WorkerChannelListInstance {
    _version: V1;
    _solution: WorkerChannelSolution;
    _uri: string;
    (sid: string): WorkerChannelContext;
    get(sid: string): WorkerChannelContext;
    /**
     * Streams WorkerChannelInstance records from the API.
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
     * @param { WorkerChannelListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: WorkerChannelInstance, done: (err?: Error) => void) => void): void;
    each(params: WorkerChannelListInstanceEachOptions, callback?: (item: WorkerChannelInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of WorkerChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: WorkerChannelPage) => any): Promise<WorkerChannelPage>;
    /**
     * Lists WorkerChannelInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { WorkerChannelListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: WorkerChannelInstance[]) => any): Promise<WorkerChannelInstance[]>;
    list(params: WorkerChannelListInstanceOptions, callback?: (error: Error | null, items: WorkerChannelInstance[]) => any): Promise<WorkerChannelInstance[]>;
    /**
     * Retrieve a single page of WorkerChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { WorkerChannelListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: WorkerChannelPage) => any): Promise<WorkerChannelPage>;
    page(params: WorkerChannelListInstancePageOptions, callback?: (error: Error | null, items: WorkerChannelPage) => any): Promise<WorkerChannelPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WorkerChannelListInstance(version: V1, workspaceSid: string, workerSid: string): WorkerChannelListInstance;
export declare class WorkerChannelPage extends Page<V1, WorkerChannelPayload, WorkerChannelResource, WorkerChannelInstance> {
    /**
     * Initialize the WorkerChannelPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: WorkerChannelSolution);
    /**
     * Build an instance of WorkerChannelInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: WorkerChannelResource): WorkerChannelInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
