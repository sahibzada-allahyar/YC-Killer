/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { ActivityListInstance } from "./workspace/activity";
import { EventListInstance } from "./workspace/event";
import { TaskListInstance } from "./workspace/task";
import { TaskChannelListInstance } from "./workspace/taskChannel";
import { TaskQueueListInstance } from "./workspace/taskQueue";
import { WorkerListInstance } from "./workspace/worker";
import { WorkflowListInstance } from "./workspace/workflow";
import { WorkspaceCumulativeStatisticsListInstance } from "./workspace/workspaceCumulativeStatistics";
import { WorkspaceRealTimeStatisticsListInstance } from "./workspace/workspaceRealTimeStatistics";
import { WorkspaceStatisticsListInstance } from "./workspace/workspaceStatistics";
export type WorkspaceQueueOrder = "FIFO" | "LIFO";
/**
 * Options to pass to update a WorkspaceInstance
 */
export interface WorkspaceContextUpdateOptions {
    /** The SID of the Activity that will be used when new Workers are created in the Workspace. */
    defaultActivitySid?: string;
    /** The URL we should call when an event occurs. See [Workspace Events](https://www.twilio.com/docs/taskrouter/api/event) for more information. This parameter supports Twilio\\\'s [Webhooks (HTTP callbacks) Connection Overrides](https://www.twilio.com/docs/usage/webhooks/webhooks-connection-overrides). */
    eventCallbackUrl?: string;
    /** The list of Workspace events for which to call event_callback_url. For example if `EventsFilter=task.created,task.canceled,worker.activity.update`, then TaskRouter will call event_callback_url only when a task is created, canceled, or a Worker activity is updated. */
    eventsFilter?: string;
    /** A descriptive string that you create to describe the Workspace resource. For example: `Sales Call Center` or `Customer Support Team`. */
    friendlyName?: string;
    /** Whether to enable multi-tasking. Can be: `true` to enable multi-tasking, or `false` to disable it. However, all workspaces should be maintained as multi-tasking. There is no default when omitting this parameter. A multi-tasking Workspace can\\\'t be updated to single-tasking unless it is not a Flex Project and another (legacy) single-tasking Workspace exists. Multi-tasking allows Workers to handle multiple Tasks simultaneously. In multi-tasking mode, each Worker can receive parallel reservations up to the per-channel maximums defined in the Workers section. In single-tasking mode (legacy mode), each Worker will only receive a new reservation when the previous task is completed. Learn more at [Multitasking](https://www.twilio.com/docs/taskrouter/multitasking). */
    multiTaskEnabled?: boolean;
    /** The SID of the Activity that will be assigned to a Worker when a Task reservation times out without a response. */
    timeoutActivitySid?: string;
    /**  */
    prioritizeQueueOrder?: WorkspaceQueueOrder;
}
/**
 * Options to pass to create a WorkspaceInstance
 */
export interface WorkspaceListInstanceCreateOptions {
    /** A descriptive string that you create to describe the Workspace resource. It can be up to 64 characters long. For example: `Customer Support` or `2014 Election Campaign`. */
    friendlyName: string;
    /** The URL we should call when an event occurs. If provided, the Workspace will publish events to this URL, for example, to collect data for reporting. See [Workspace Events](https://www.twilio.com/docs/taskrouter/api/event) for more information. This parameter supports Twilio\\\'s [Webhooks (HTTP callbacks) Connection Overrides](https://www.twilio.com/docs/usage/webhooks/webhooks-connection-overrides). */
    eventCallbackUrl?: string;
    /** The list of Workspace events for which to call event_callback_url. For example, if `EventsFilter=task.created, task.canceled, worker.activity.update`, then TaskRouter will call event_callback_url only when a task is created, canceled, or a Worker activity is updated. */
    eventsFilter?: string;
    /** Whether to enable multi-tasking. Can be: `true` to enable multi-tasking, or `false` to disable it. However, all workspaces should be created as multi-tasking. The default is `true`. Multi-tasking allows Workers to handle multiple Tasks simultaneously. When enabled (`true`), each Worker can receive parallel reservations up to the per-channel maximums defined in the Workers section. In single-tasking mode (legacy mode), each Worker will only receive a new reservation when the previous task is completed. Learn more at [Multitasking](https://www.twilio.com/docs/taskrouter/multitasking). */
    multiTaskEnabled?: boolean;
    /** An available template name. Can be: `NONE` or `FIFO` and the default is `NONE`. Pre-configures the Workspace with the Workflow and Activities specified in the template. `NONE` will create a Workspace with only a set of default activities. `FIFO` will configure TaskRouter with a set of default activities and a single TaskQueue for first-in, first-out distribution, which can be useful when you are getting started with TaskRouter. */
    template?: string;
    /**  */
    prioritizeQueueOrder?: WorkspaceQueueOrder;
}
/**
 * Options to pass to each
 */
export interface WorkspaceListInstanceEachOptions {
    /** The `friendly_name` of the Workspace resources to read. For example `Customer Support` or `2014 Election Campaign`. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: WorkspaceInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface WorkspaceListInstanceOptions {
    /** The `friendly_name` of the Workspace resources to read. For example `Customer Support` or `2014 Election Campaign`. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface WorkspaceListInstancePageOptions {
    /** The `friendly_name` of the Workspace resources to read. For example `Customer Support` or `2014 Election Campaign`. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface WorkspaceContext {
    activities: ActivityListInstance;
    events: EventListInstance;
    tasks: TaskListInstance;
    taskChannels: TaskChannelListInstance;
    taskQueues: TaskQueueListInstance;
    workers: WorkerListInstance;
    workflows: WorkflowListInstance;
    cumulativeStatistics: WorkspaceCumulativeStatisticsListInstance;
    realTimeStatistics: WorkspaceRealTimeStatisticsListInstance;
    statistics: WorkspaceStatisticsListInstance;
    /**
     * Remove a WorkspaceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a WorkspaceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkspaceInstance) => any): Promise<WorkspaceInstance>;
    /**
     * Update a WorkspaceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceInstance
     */
    update(callback?: (error: Error | null, item?: WorkspaceInstance) => any): Promise<WorkspaceInstance>;
    /**
     * Update a WorkspaceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceInstance
     */
    update(params: WorkspaceContextUpdateOptions, callback?: (error: Error | null, item?: WorkspaceInstance) => any): Promise<WorkspaceInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WorkspaceContextSolution {
    sid: string;
}
export declare class WorkspaceContextImpl implements WorkspaceContext {
    protected _version: V1;
    protected _solution: WorkspaceContextSolution;
    protected _uri: string;
    protected _activities?: ActivityListInstance;
    protected _events?: EventListInstance;
    protected _tasks?: TaskListInstance;
    protected _taskChannels?: TaskChannelListInstance;
    protected _taskQueues?: TaskQueueListInstance;
    protected _workers?: WorkerListInstance;
    protected _workflows?: WorkflowListInstance;
    protected _cumulativeStatistics?: WorkspaceCumulativeStatisticsListInstance;
    protected _realTimeStatistics?: WorkspaceRealTimeStatisticsListInstance;
    protected _statistics?: WorkspaceStatisticsListInstance;
    constructor(_version: V1, sid: string);
    get activities(): ActivityListInstance;
    get events(): EventListInstance;
    get tasks(): TaskListInstance;
    get taskChannels(): TaskChannelListInstance;
    get taskQueues(): TaskQueueListInstance;
    get workers(): WorkerListInstance;
    get workflows(): WorkflowListInstance;
    get cumulativeStatistics(): WorkspaceCumulativeStatisticsListInstance;
    get realTimeStatistics(): WorkspaceRealTimeStatisticsListInstance;
    get statistics(): WorkspaceStatisticsListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: WorkspaceInstance) => any): Promise<WorkspaceInstance>;
    update(params?: WorkspaceContextUpdateOptions | ((error: Error | null, item?: WorkspaceInstance) => any), callback?: (error: Error | null, item?: WorkspaceInstance) => any): Promise<WorkspaceInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WorkspaceContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WorkspacePayload extends TwilioResponsePayload {
    workspaces: WorkspaceResource[];
}
interface WorkspaceResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    default_activity_name: string;
    default_activity_sid: string;
    event_callback_url: string;
    events_filter: string;
    friendly_name: string;
    multi_task_enabled: boolean;
    sid: string;
    timeout_activity_name: string;
    timeout_activity_sid: string;
    prioritize_queue_order: WorkspaceQueueOrder;
    url: string;
    links: Record<string, string>;
}
export declare class WorkspaceInstance {
    protected _version: V1;
    protected _solution: WorkspaceContextSolution;
    protected _context?: WorkspaceContext;
    constructor(_version: V1, payload: WorkspaceResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Workspace resource.
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
     * The name of the default activity.
     */
    defaultActivityName: string;
    /**
     * The SID of the Activity that will be used when new Workers are created in the Workspace.
     */
    defaultActivitySid: string;
    /**
     * The URL we call when an event occurs. If provided, the Workspace will publish events to this URL, for example, to collect data for reporting. See [Workspace Events](https://www.twilio.com/docs/taskrouter/api/event) for more information. This parameter supports Twilio\'s [Webhooks (HTTP callbacks) Connection Overrides](https://www.twilio.com/docs/usage/webhooks/webhooks-connection-overrides).
     */
    eventCallbackUrl: string;
    /**
     * The list of Workspace events for which to call `event_callback_url`. For example, if `EventsFilter=task.created, task.canceled, worker.activity.update`, then TaskRouter will call event_callback_url only when a task is created, canceled, or a Worker activity is updated.
     */
    eventsFilter: string;
    /**
     * The string that you assigned to describe the Workspace resource. For example `Customer Support` or `2014 Election Campaign`.
     */
    friendlyName: string;
    /**
     * Whether multi-tasking is enabled. The default is `true`, which enables multi-tasking. Multi-tasking allows Workers to handle multiple Tasks simultaneously. When enabled (`true`), each Worker can receive parallel reservations up to the per-channel maximums defined in the Workers section. In single-tasking each Worker would only receive a new reservation when the previous task is completed. Learn more at [Multitasking](https://www.twilio.com/docs/taskrouter/multitasking).
     */
    multiTaskEnabled: boolean;
    /**
     * The unique string that we created to identify the Workspace resource.
     */
    sid: string;
    /**
     * The name of the timeout activity.
     */
    timeoutActivityName: string;
    /**
     * The SID of the Activity that will be assigned to a Worker when a Task reservation times out without a response.
     */
    timeoutActivitySid: string;
    prioritizeQueueOrder: WorkspaceQueueOrder;
    /**
     * The absolute URL of the Workspace resource.
     */
    url: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a WorkspaceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a WorkspaceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkspaceInstance) => any): Promise<WorkspaceInstance>;
    /**
     * Update a WorkspaceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceInstance
     */
    update(callback?: (error: Error | null, item?: WorkspaceInstance) => any): Promise<WorkspaceInstance>;
    /**
     * Update a WorkspaceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceInstance
     */
    update(params: WorkspaceContextUpdateOptions, callback?: (error: Error | null, item?: WorkspaceInstance) => any): Promise<WorkspaceInstance>;
    /**
     * Access the activities.
     */
    activities(): ActivityListInstance;
    /**
     * Access the events.
     */
    events(): EventListInstance;
    /**
     * Access the tasks.
     */
    tasks(): TaskListInstance;
    /**
     * Access the taskChannels.
     */
    taskChannels(): TaskChannelListInstance;
    /**
     * Access the taskQueues.
     */
    taskQueues(): TaskQueueListInstance;
    /**
     * Access the workers.
     */
    workers(): WorkerListInstance;
    /**
     * Access the workflows.
     */
    workflows(): WorkflowListInstance;
    /**
     * Access the cumulativeStatistics.
     */
    cumulativeStatistics(): WorkspaceCumulativeStatisticsListInstance;
    /**
     * Access the realTimeStatistics.
     */
    realTimeStatistics(): WorkspaceRealTimeStatisticsListInstance;
    /**
     * Access the statistics.
     */
    statistics(): WorkspaceStatisticsListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        defaultActivityName: string;
        defaultActivitySid: string;
        eventCallbackUrl: string;
        eventsFilter: string;
        friendlyName: string;
        multiTaskEnabled: boolean;
        sid: string;
        timeoutActivityName: string;
        timeoutActivitySid: string;
        prioritizeQueueOrder: WorkspaceQueueOrder;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WorkspaceSolution {
}
export interface WorkspaceListInstance {
    _version: V1;
    _solution: WorkspaceSolution;
    _uri: string;
    (sid: string): WorkspaceContext;
    get(sid: string): WorkspaceContext;
    /**
     * Create a WorkspaceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceInstance
     */
    create(params: WorkspaceListInstanceCreateOptions, callback?: (error: Error | null, item?: WorkspaceInstance) => any): Promise<WorkspaceInstance>;
    /**
     * Streams WorkspaceInstance records from the API.
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
     * @param { WorkspaceListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: WorkspaceInstance, done: (err?: Error) => void) => void): void;
    each(params: WorkspaceListInstanceEachOptions, callback?: (item: WorkspaceInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of WorkspaceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: WorkspacePage) => any): Promise<WorkspacePage>;
    /**
     * Lists WorkspaceInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { WorkspaceListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: WorkspaceInstance[]) => any): Promise<WorkspaceInstance[]>;
    list(params: WorkspaceListInstanceOptions, callback?: (error: Error | null, items: WorkspaceInstance[]) => any): Promise<WorkspaceInstance[]>;
    /**
     * Retrieve a single page of WorkspaceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { WorkspaceListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: WorkspacePage) => any): Promise<WorkspacePage>;
    page(params: WorkspaceListInstancePageOptions, callback?: (error: Error | null, items: WorkspacePage) => any): Promise<WorkspacePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WorkspaceListInstance(version: V1): WorkspaceListInstance;
export declare class WorkspacePage extends Page<V1, WorkspacePayload, WorkspaceResource, WorkspaceInstance> {
    /**
     * Initialize the WorkspacePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: WorkspaceSolution);
    /**
     * Build an instance of WorkspaceInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: WorkspaceResource): WorkspaceInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
