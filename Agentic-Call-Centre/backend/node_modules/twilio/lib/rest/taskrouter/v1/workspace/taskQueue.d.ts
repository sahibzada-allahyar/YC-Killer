/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { TaskQueueCumulativeStatisticsListInstance } from "./taskQueue/taskQueueCumulativeStatistics";
import { TaskQueueRealTimeStatisticsListInstance } from "./taskQueue/taskQueueRealTimeStatistics";
import { TaskQueueStatisticsListInstance } from "./taskQueue/taskQueueStatistics";
import { TaskQueuesStatisticsListInstance } from "./taskQueue/taskQueuesStatistics";
export type TaskQueueTaskOrder = "FIFO" | "LIFO";
/**
 * Options to pass to update a TaskQueueInstance
 */
export interface TaskQueueContextUpdateOptions {
    /** A descriptive string that you create to describe the TaskQueue. For example `Support-Tier 1`, `Sales`, or `Escalation`. */
    friendlyName?: string;
    /** A string describing the Worker selection criteria for any Tasks that enter the TaskQueue. For example \\\'\\\"language\\\" == \\\"spanish\\\"\\\' If no TargetWorkers parameter is provided, Tasks will wait in the queue until they are either deleted or moved to another queue. Additional examples on how to describing Worker selection criteria below. */
    targetWorkers?: string;
    /** The SID of the Activity to assign Workers when a task is reserved for them. */
    reservationActivitySid?: string;
    /** The SID of the Activity to assign Workers when a task is assigned for them. */
    assignmentActivitySid?: string;
    /** The maximum number of Workers to create reservations for the assignment of a task while in the queue. Maximum of 50. */
    maxReservedWorkers?: number;
    /**  */
    taskOrder?: TaskQueueTaskOrder;
}
/**
 * Options to pass to create a TaskQueueInstance
 */
export interface TaskQueueListInstanceCreateOptions {
    /** A descriptive string that you create to describe the TaskQueue. For example `Support-Tier 1`, `Sales`, or `Escalation`. */
    friendlyName: string;
    /** A string that describes the Worker selection criteria for any Tasks that enter the TaskQueue. For example, `\\\'\\\"language\\\" == \\\"spanish\\\"\\\'`. The default value is `1==1`. If this value is empty, Tasks will wait in the TaskQueue until they are deleted or moved to another TaskQueue. For more information about Worker selection, see [Describing Worker selection criteria](https://www.twilio.com/docs/taskrouter/api/taskqueues#target-workers). */
    targetWorkers?: string;
    /** The maximum number of Workers to reserve for the assignment of a Task in the queue. Can be an integer between 1 and 50, inclusive and defaults to 1. */
    maxReservedWorkers?: number;
    /**  */
    taskOrder?: TaskQueueTaskOrder;
    /** The SID of the Activity to assign Workers when a task is reserved for them. */
    reservationActivitySid?: string;
    /** The SID of the Activity to assign Workers when a task is assigned to them. */
    assignmentActivitySid?: string;
}
/**
 * Options to pass to each
 */
export interface TaskQueueListInstanceEachOptions {
    /** The `friendly_name` of the TaskQueue resources to read. */
    friendlyName?: string;
    /** The attributes of the Workers to read. Returns the TaskQueues with Workers that match the attributes specified in this parameter. */
    evaluateWorkerAttributes?: string;
    /** The SID of the Worker with the TaskQueue resources to read. */
    workerSid?: string;
    /** Sorting parameter for TaskQueues */
    ordering?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TaskQueueInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TaskQueueListInstanceOptions {
    /** The `friendly_name` of the TaskQueue resources to read. */
    friendlyName?: string;
    /** The attributes of the Workers to read. Returns the TaskQueues with Workers that match the attributes specified in this parameter. */
    evaluateWorkerAttributes?: string;
    /** The SID of the Worker with the TaskQueue resources to read. */
    workerSid?: string;
    /** Sorting parameter for TaskQueues */
    ordering?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TaskQueueListInstancePageOptions {
    /** The `friendly_name` of the TaskQueue resources to read. */
    friendlyName?: string;
    /** The attributes of the Workers to read. Returns the TaskQueues with Workers that match the attributes specified in this parameter. */
    evaluateWorkerAttributes?: string;
    /** The SID of the Worker with the TaskQueue resources to read. */
    workerSid?: string;
    /** Sorting parameter for TaskQueues */
    ordering?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TaskQueueContext {
    cumulativeStatistics: TaskQueueCumulativeStatisticsListInstance;
    realTimeStatistics: TaskQueueRealTimeStatisticsListInstance;
    statistics: TaskQueueStatisticsListInstance;
    /**
     * Remove a TaskQueueInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TaskQueueInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueInstance
     */
    fetch(callback?: (error: Error | null, item?: TaskQueueInstance) => any): Promise<TaskQueueInstance>;
    /**
     * Update a TaskQueueInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueInstance
     */
    update(callback?: (error: Error | null, item?: TaskQueueInstance) => any): Promise<TaskQueueInstance>;
    /**
     * Update a TaskQueueInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueInstance
     */
    update(params: TaskQueueContextUpdateOptions, callback?: (error: Error | null, item?: TaskQueueInstance) => any): Promise<TaskQueueInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TaskQueueContextSolution {
    workspaceSid: string;
    sid: string;
}
export declare class TaskQueueContextImpl implements TaskQueueContext {
    protected _version: V1;
    protected _solution: TaskQueueContextSolution;
    protected _uri: string;
    protected _cumulativeStatistics?: TaskQueueCumulativeStatisticsListInstance;
    protected _realTimeStatistics?: TaskQueueRealTimeStatisticsListInstance;
    protected _statistics?: TaskQueueStatisticsListInstance;
    constructor(_version: V1, workspaceSid: string, sid: string);
    get cumulativeStatistics(): TaskQueueCumulativeStatisticsListInstance;
    get realTimeStatistics(): TaskQueueRealTimeStatisticsListInstance;
    get statistics(): TaskQueueStatisticsListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: TaskQueueInstance) => any): Promise<TaskQueueInstance>;
    update(params?: TaskQueueContextUpdateOptions | ((error: Error | null, item?: TaskQueueInstance) => any), callback?: (error: Error | null, item?: TaskQueueInstance) => any): Promise<TaskQueueInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TaskQueueContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TaskQueuePayload extends TwilioResponsePayload {
    task_queues: TaskQueueResource[];
}
interface TaskQueueResource {
    account_sid: string;
    assignment_activity_sid: string;
    assignment_activity_name: string;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    max_reserved_workers: number;
    reservation_activity_sid: string;
    reservation_activity_name: string;
    sid: string;
    target_workers: string;
    task_order: TaskQueueTaskOrder;
    url: string;
    workspace_sid: string;
    links: Record<string, string>;
}
export declare class TaskQueueInstance {
    protected _version: V1;
    protected _solution: TaskQueueContextSolution;
    protected _context?: TaskQueueContext;
    constructor(_version: V1, payload: TaskQueueResource, workspaceSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the TaskQueue resource.
     */
    accountSid: string;
    /**
     * The SID of the Activity to assign Workers when a task is assigned for them.
     */
    assignmentActivitySid: string;
    /**
     * The name of the Activity to assign Workers when a task is assigned for them.
     */
    assignmentActivityName: string;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The maximum number of Workers to reserve for the assignment of a task in the queue. Can be an integer between 1 and 50, inclusive and defaults to 1.
     */
    maxReservedWorkers: number;
    /**
     * The SID of the Activity to assign Workers once a task is reserved for them.
     */
    reservationActivitySid: string;
    /**
     * The name of the Activity to assign Workers once a task is reserved for them.
     */
    reservationActivityName: string;
    /**
     * The unique string that we created to identify the TaskQueue resource.
     */
    sid: string;
    /**
     * A string describing the Worker selection criteria for any Tasks that enter the TaskQueue. For example `\'\"language\" == \"spanish\"\'` If no TargetWorkers parameter is provided, Tasks will wait in the TaskQueue until they are either deleted or moved to another TaskQueue. Additional examples on how to describing Worker selection criteria below. Defaults to 1==1.
     */
    targetWorkers: string;
    taskOrder: TaskQueueTaskOrder;
    /**
     * The absolute URL of the TaskQueue resource.
     */
    url: string;
    /**
     * The SID of the Workspace that contains the TaskQueue.
     */
    workspaceSid: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a TaskQueueInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TaskQueueInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueInstance
     */
    fetch(callback?: (error: Error | null, item?: TaskQueueInstance) => any): Promise<TaskQueueInstance>;
    /**
     * Update a TaskQueueInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueInstance
     */
    update(callback?: (error: Error | null, item?: TaskQueueInstance) => any): Promise<TaskQueueInstance>;
    /**
     * Update a TaskQueueInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueInstance
     */
    update(params: TaskQueueContextUpdateOptions, callback?: (error: Error | null, item?: TaskQueueInstance) => any): Promise<TaskQueueInstance>;
    /**
     * Access the cumulativeStatistics.
     */
    cumulativeStatistics(): TaskQueueCumulativeStatisticsListInstance;
    /**
     * Access the realTimeStatistics.
     */
    realTimeStatistics(): TaskQueueRealTimeStatisticsListInstance;
    /**
     * Access the statistics.
     */
    statistics(): TaskQueueStatisticsListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        assignmentActivitySid: string;
        assignmentActivityName: string;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        maxReservedWorkers: number;
        reservationActivitySid: string;
        reservationActivityName: string;
        sid: string;
        targetWorkers: string;
        taskOrder: TaskQueueTaskOrder;
        url: string;
        workspaceSid: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TaskQueueSolution {
    workspaceSid: string;
}
export interface TaskQueueListInstance {
    _version: V1;
    _solution: TaskQueueSolution;
    _uri: string;
    (sid: string): TaskQueueContext;
    get(sid: string): TaskQueueContext;
    _statistics?: TaskQueuesStatisticsListInstance;
    statistics: TaskQueuesStatisticsListInstance;
    /**
     * Create a TaskQueueInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueInstance
     */
    create(params: TaskQueueListInstanceCreateOptions, callback?: (error: Error | null, item?: TaskQueueInstance) => any): Promise<TaskQueueInstance>;
    /**
     * Streams TaskQueueInstance records from the API.
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
     * @param { TaskQueueListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TaskQueueInstance, done: (err?: Error) => void) => void): void;
    each(params: TaskQueueListInstanceEachOptions, callback?: (item: TaskQueueInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TaskQueueInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TaskQueuePage) => any): Promise<TaskQueuePage>;
    /**
     * Lists TaskQueueInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TaskQueueListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TaskQueueInstance[]) => any): Promise<TaskQueueInstance[]>;
    list(params: TaskQueueListInstanceOptions, callback?: (error: Error | null, items: TaskQueueInstance[]) => any): Promise<TaskQueueInstance[]>;
    /**
     * Retrieve a single page of TaskQueueInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TaskQueueListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TaskQueuePage) => any): Promise<TaskQueuePage>;
    page(params: TaskQueueListInstancePageOptions, callback?: (error: Error | null, items: TaskQueuePage) => any): Promise<TaskQueuePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TaskQueueListInstance(version: V1, workspaceSid: string): TaskQueueListInstance;
export declare class TaskQueuePage extends Page<V1, TaskQueuePayload, TaskQueueResource, TaskQueueInstance> {
    /**
     * Initialize the TaskQueuePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: TaskQueueSolution);
    /**
     * Build an instance of TaskQueueInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TaskQueueResource): TaskQueueInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
