/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { ReservationListInstance } from "./task/reservation";
export type TaskStatus = "pending" | "reserved" | "assigned" | "canceled" | "completed" | "wrapping";
/**
 * Options to pass to remove a TaskInstance
 */
export interface TaskContextRemoveOptions {
    /** If provided, deletes this Task if (and only if) the [ETag](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag) header of the Task matches the provided value. This matches the semantics of (and is implemented with) the HTTP [If-Match header](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/If-Match). */
    ifMatch?: string;
}
/**
 * Options to pass to update a TaskInstance
 */
export interface TaskContextUpdateOptions {
    /** If provided, applies this mutation if (and only if) the [ETag](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag) header of the Task matches the provided value. This matches the semantics of (and is implemented with) the HTTP [If-Match header](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/If-Match). */
    ifMatch?: string;
    /** The JSON string that describes the custom attributes of the task. */
    attributes?: string;
    /**  */
    assignmentStatus?: TaskStatus;
    /** The reason that the Task was canceled or completed. This parameter is required only if the Task is canceled or completed. Setting this value queues the task for deletion and logs the reason. */
    reason?: string;
    /** The Task\\\'s new priority value. When supplied, the Task takes on the specified priority unless it matches a Workflow Target with a Priority set. Value can be 0 to 2^31^ (2,147,483,647). */
    priority?: number;
    /** When MultiTasking is enabled, specify the TaskChannel with the task to update. Can be the TaskChannel\\\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
    /** The task\\\'s new virtual start time value. When supplied, the Task takes on the specified virtual start time. Value can\\\'t be in the future. */
    virtualStartTime?: Date;
}
/**
 * Options to pass to create a TaskInstance
 */
export interface TaskListInstanceCreateOptions {
    /** The amount of time in seconds the new task can live before being assigned. Can be up to a maximum of 2 weeks (1,209,600 seconds). The default value is 24 hours (86,400 seconds). On timeout, the `task.canceled` event will fire with description `Task TTL Exceeded`. */
    timeout?: number;
    /** The priority to assign the new task and override the default. When supplied, the new Task will have this priority unless it matches a Workflow Target with a Priority set. When not supplied, the new Task will have the priority of the matching Workflow Target. Value can be 0 to 2^31^ (2,147,483,647). */
    priority?: number;
    /** When MultiTasking is enabled, specify the TaskChannel by passing either its `unique_name` or `sid`. Default value is `default`. */
    taskChannel?: string;
    /** The SID of the Workflow that you would like to handle routing for the new Task. If there is only one Workflow defined for the Workspace that you are posting the new task to, this parameter is optional. */
    workflowSid?: string;
    /** A URL-encoded JSON string with the attributes of the new task. This value is passed to the Workflow\\\'s `assignment_callback_url` when the Task is assigned to a Worker. For example: `{ \\\"task_type\\\": \\\"call\\\", \\\"twilio_call_sid\\\": \\\"CAxxx\\\", \\\"customer_ticket_number\\\": \\\"12345\\\" }`. */
    attributes?: string;
    /** The virtual start time to assign the new task and override the default. When supplied, the new task will have this virtual start time. When not supplied, the new task will have the virtual start time equal to `date_created`. Value can\\\'t be in the future. */
    virtualStartTime?: Date;
}
/**
 * Options to pass to each
 */
export interface TaskListInstanceEachOptions {
    /** The priority value of the Tasks to read. Returns the list of all Tasks in the Workspace with the specified priority. */
    priority?: number;
    /** The `assignment_status` of the Tasks you want to read. Can be: `pending`, `reserved`, `assigned`, `canceled`, `wrapping`, or `completed`. Returns all Tasks in the Workspace with the specified `assignment_status`. */
    assignmentStatus?: Array<string>;
    /** The SID of the Workflow with the Tasks to read. Returns the Tasks controlled by the Workflow identified by this SID. */
    workflowSid?: string;
    /** The friendly name of the Workflow with the Tasks to read. Returns the Tasks controlled by the Workflow identified by this friendly name. */
    workflowName?: string;
    /** The SID of the TaskQueue with the Tasks to read. Returns the Tasks waiting in the TaskQueue identified by this SID. */
    taskQueueSid?: string;
    /** The `friendly_name` of the TaskQueue with the Tasks to read. Returns the Tasks waiting in the TaskQueue identified by this friendly name. */
    taskQueueName?: string;
    /** The attributes of the Tasks to read. Returns the Tasks that match the attributes specified in this parameter. */
    evaluateTaskAttributes?: string;
    /** How to order the returned Task resources. By default, Tasks are sorted by ascending DateCreated. This value is specified as: `Attribute:Order`, where `Attribute` can be either `DateCreated`, `Priority`, or `VirtualStartTime` and `Order` can be either `asc` or `desc`. For example, `Priority:desc` returns Tasks ordered in descending order of their Priority. Pairings of sort orders can be specified in a comma-separated list such as `Priority:desc,DateCreated:asc`, which returns the Tasks in descending Priority order and ascending DateCreated Order. The only ordering pairing not allowed is DateCreated and VirtualStartTime. */
    ordering?: string;
    /** Whether to read Tasks with Add-ons. If `true`, returns only Tasks with Add-ons. If `false`, returns only Tasks without Add-ons. */
    hasAddons?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TaskInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TaskListInstanceOptions {
    /** The priority value of the Tasks to read. Returns the list of all Tasks in the Workspace with the specified priority. */
    priority?: number;
    /** The `assignment_status` of the Tasks you want to read. Can be: `pending`, `reserved`, `assigned`, `canceled`, `wrapping`, or `completed`. Returns all Tasks in the Workspace with the specified `assignment_status`. */
    assignmentStatus?: Array<string>;
    /** The SID of the Workflow with the Tasks to read. Returns the Tasks controlled by the Workflow identified by this SID. */
    workflowSid?: string;
    /** The friendly name of the Workflow with the Tasks to read. Returns the Tasks controlled by the Workflow identified by this friendly name. */
    workflowName?: string;
    /** The SID of the TaskQueue with the Tasks to read. Returns the Tasks waiting in the TaskQueue identified by this SID. */
    taskQueueSid?: string;
    /** The `friendly_name` of the TaskQueue with the Tasks to read. Returns the Tasks waiting in the TaskQueue identified by this friendly name. */
    taskQueueName?: string;
    /** The attributes of the Tasks to read. Returns the Tasks that match the attributes specified in this parameter. */
    evaluateTaskAttributes?: string;
    /** How to order the returned Task resources. By default, Tasks are sorted by ascending DateCreated. This value is specified as: `Attribute:Order`, where `Attribute` can be either `DateCreated`, `Priority`, or `VirtualStartTime` and `Order` can be either `asc` or `desc`. For example, `Priority:desc` returns Tasks ordered in descending order of their Priority. Pairings of sort orders can be specified in a comma-separated list such as `Priority:desc,DateCreated:asc`, which returns the Tasks in descending Priority order and ascending DateCreated Order. The only ordering pairing not allowed is DateCreated and VirtualStartTime. */
    ordering?: string;
    /** Whether to read Tasks with Add-ons. If `true`, returns only Tasks with Add-ons. If `false`, returns only Tasks without Add-ons. */
    hasAddons?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TaskListInstancePageOptions {
    /** The priority value of the Tasks to read. Returns the list of all Tasks in the Workspace with the specified priority. */
    priority?: number;
    /** The `assignment_status` of the Tasks you want to read. Can be: `pending`, `reserved`, `assigned`, `canceled`, `wrapping`, or `completed`. Returns all Tasks in the Workspace with the specified `assignment_status`. */
    assignmentStatus?: Array<string>;
    /** The SID of the Workflow with the Tasks to read. Returns the Tasks controlled by the Workflow identified by this SID. */
    workflowSid?: string;
    /** The friendly name of the Workflow with the Tasks to read. Returns the Tasks controlled by the Workflow identified by this friendly name. */
    workflowName?: string;
    /** The SID of the TaskQueue with the Tasks to read. Returns the Tasks waiting in the TaskQueue identified by this SID. */
    taskQueueSid?: string;
    /** The `friendly_name` of the TaskQueue with the Tasks to read. Returns the Tasks waiting in the TaskQueue identified by this friendly name. */
    taskQueueName?: string;
    /** The attributes of the Tasks to read. Returns the Tasks that match the attributes specified in this parameter. */
    evaluateTaskAttributes?: string;
    /** How to order the returned Task resources. By default, Tasks are sorted by ascending DateCreated. This value is specified as: `Attribute:Order`, where `Attribute` can be either `DateCreated`, `Priority`, or `VirtualStartTime` and `Order` can be either `asc` or `desc`. For example, `Priority:desc` returns Tasks ordered in descending order of their Priority. Pairings of sort orders can be specified in a comma-separated list such as `Priority:desc,DateCreated:asc`, which returns the Tasks in descending Priority order and ascending DateCreated Order. The only ordering pairing not allowed is DateCreated and VirtualStartTime. */
    ordering?: string;
    /** Whether to read Tasks with Add-ons. If `true`, returns only Tasks with Add-ons. If `false`, returns only Tasks without Add-ons. */
    hasAddons?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TaskContext {
    reservations: ReservationListInstance;
    /**
     * Remove a TaskInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a TaskInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskInstance
     */
    remove(params: TaskContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TaskInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskInstance
     */
    fetch(callback?: (error: Error | null, item?: TaskInstance) => any): Promise<TaskInstance>;
    /**
     * Update a TaskInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskInstance
     */
    update(callback?: (error: Error | null, item?: TaskInstance) => any): Promise<TaskInstance>;
    /**
     * Update a TaskInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskInstance
     */
    update(params: TaskContextUpdateOptions, callback?: (error: Error | null, item?: TaskInstance) => any): Promise<TaskInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TaskContextSolution {
    workspaceSid: string;
    sid: string;
}
export declare class TaskContextImpl implements TaskContext {
    protected _version: V1;
    protected _solution: TaskContextSolution;
    protected _uri: string;
    protected _reservations?: ReservationListInstance;
    constructor(_version: V1, workspaceSid: string, sid: string);
    get reservations(): ReservationListInstance;
    remove(params?: TaskContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: TaskInstance) => any): Promise<TaskInstance>;
    update(params?: TaskContextUpdateOptions | ((error: Error | null, item?: TaskInstance) => any), callback?: (error: Error | null, item?: TaskInstance) => any): Promise<TaskInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TaskContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TaskPayload extends TwilioResponsePayload {
    tasks: TaskResource[];
}
interface TaskResource {
    account_sid: string;
    age: number;
    assignment_status: TaskStatus;
    attributes: string;
    addons: string;
    date_created: Date;
    date_updated: Date;
    task_queue_entered_date: Date;
    priority: number;
    reason: string;
    sid: string;
    task_queue_sid: string;
    task_queue_friendly_name: string;
    task_channel_sid: string;
    task_channel_unique_name: string;
    timeout: number;
    workflow_sid: string;
    workflow_friendly_name: string;
    workspace_sid: string;
    url: string;
    links: Record<string, string>;
    virtual_start_time: Date;
}
export declare class TaskInstance {
    protected _version: V1;
    protected _solution: TaskContextSolution;
    protected _context?: TaskContext;
    constructor(_version: V1, payload: TaskResource, workspaceSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Task resource.
     */
    accountSid: string;
    /**
     * The number of seconds since the Task was created.
     */
    age: number;
    assignmentStatus: TaskStatus;
    /**
     * The JSON string with custom attributes of the work. **Note** If this property has been assigned a value, it will only be displayed in FETCH action that returns a single resource. Otherwise, it will be null.
     */
    attributes: string;
    /**
     * An object that contains the [Add-on](https://www.twilio.com/docs/add-ons) data for all installed Add-ons.
     */
    addons: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The date and time in GMT when the Task entered the TaskQueue, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    taskQueueEnteredDate: Date;
    /**
     * The current priority score of the Task as assigned to a Worker by the workflow. Tasks with higher priority values will be assigned before Tasks with lower values.
     */
    priority: number;
    /**
     * The reason the Task was canceled or completed, if applicable.
     */
    reason: string;
    /**
     * The unique string that we created to identify the Task resource.
     */
    sid: string;
    /**
     * The SID of the TaskQueue.
     */
    taskQueueSid: string;
    /**
     * The friendly name of the TaskQueue.
     */
    taskQueueFriendlyName: string;
    /**
     * The SID of the TaskChannel.
     */
    taskChannelSid: string;
    /**
     * The unique name of the TaskChannel.
     */
    taskChannelUniqueName: string;
    /**
     * The amount of time in seconds that the Task can live before being assigned.
     */
    timeout: number;
    /**
     * The SID of the Workflow that is controlling the Task.
     */
    workflowSid: string;
    /**
     * The friendly name of the Workflow that is controlling the Task.
     */
    workflowFriendlyName: string;
    /**
     * The SID of the Workspace that contains the Task.
     */
    workspaceSid: string;
    /**
     * The absolute URL of the Task resource.
     */
    url: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    /**
     * The date and time in GMT indicating the ordering for routing of the Task specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    virtualStartTime: Date;
    private get _proxy();
    /**
     * Remove a TaskInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a TaskInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskInstance
     */
    remove(params: TaskContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a TaskInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskInstance
     */
    fetch(callback?: (error: Error | null, item?: TaskInstance) => any): Promise<TaskInstance>;
    /**
     * Update a TaskInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskInstance
     */
    update(callback?: (error: Error | null, item?: TaskInstance) => any): Promise<TaskInstance>;
    /**
     * Update a TaskInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskInstance
     */
    update(params: TaskContextUpdateOptions, callback?: (error: Error | null, item?: TaskInstance) => any): Promise<TaskInstance>;
    /**
     * Access the reservations.
     */
    reservations(): ReservationListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        age: number;
        assignmentStatus: TaskStatus;
        attributes: string;
        addons: string;
        dateCreated: Date;
        dateUpdated: Date;
        taskQueueEnteredDate: Date;
        priority: number;
        reason: string;
        sid: string;
        taskQueueSid: string;
        taskQueueFriendlyName: string;
        taskChannelSid: string;
        taskChannelUniqueName: string;
        timeout: number;
        workflowSid: string;
        workflowFriendlyName: string;
        workspaceSid: string;
        url: string;
        links: Record<string, string>;
        virtualStartTime: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TaskSolution {
    workspaceSid: string;
}
export interface TaskListInstance {
    _version: V1;
    _solution: TaskSolution;
    _uri: string;
    (sid: string): TaskContext;
    get(sid: string): TaskContext;
    /**
     * Create a TaskInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskInstance
     */
    create(callback?: (error: Error | null, item?: TaskInstance) => any): Promise<TaskInstance>;
    /**
     * Create a TaskInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskInstance
     */
    create(params: TaskListInstanceCreateOptions, callback?: (error: Error | null, item?: TaskInstance) => any): Promise<TaskInstance>;
    /**
     * Streams TaskInstance records from the API.
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
     * @param { TaskListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TaskInstance, done: (err?: Error) => void) => void): void;
    each(params: TaskListInstanceEachOptions, callback?: (item: TaskInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TaskInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TaskPage) => any): Promise<TaskPage>;
    /**
     * Lists TaskInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TaskListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TaskInstance[]) => any): Promise<TaskInstance[]>;
    list(params: TaskListInstanceOptions, callback?: (error: Error | null, items: TaskInstance[]) => any): Promise<TaskInstance[]>;
    /**
     * Retrieve a single page of TaskInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TaskListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TaskPage) => any): Promise<TaskPage>;
    page(params: TaskListInstancePageOptions, callback?: (error: Error | null, items: TaskPage) => any): Promise<TaskPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TaskListInstance(version: V1, workspaceSid: string): TaskListInstance;
export declare class TaskPage extends Page<V1, TaskPayload, TaskResource, TaskInstance> {
    /**
     * Initialize the TaskPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: TaskSolution);
    /**
     * Build an instance of TaskInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TaskResource): TaskInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
