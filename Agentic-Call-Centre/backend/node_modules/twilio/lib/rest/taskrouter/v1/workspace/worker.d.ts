/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { ReservationListInstance } from "./worker/reservation";
import { WorkerChannelListInstance } from "./worker/workerChannel";
import { WorkerStatisticsListInstance } from "./worker/workerStatistics";
import { WorkersCumulativeStatisticsListInstance } from "./worker/workersCumulativeStatistics";
import { WorkersRealTimeStatisticsListInstance } from "./worker/workersRealTimeStatistics";
import { WorkersStatisticsListInstance } from "./worker/workersStatistics";
/**
 * Options to pass to remove a WorkerInstance
 */
export interface WorkerContextRemoveOptions {
    /** The If-Match HTTP request header */
    ifMatch?: string;
}
/**
 * Options to pass to update a WorkerInstance
 */
export interface WorkerContextUpdateOptions {
    /** The If-Match HTTP request header */
    ifMatch?: string;
    /** The SID of a valid Activity that will describe the Worker\\\'s initial state. See [Activities](https://www.twilio.com/docs/taskrouter/api/activity) for more information. */
    activitySid?: string;
    /** The JSON string that describes the Worker. For example: `{ \\\"email\\\": \\\"Bob@example.com\\\", \\\"phone\\\": \\\"+5095551234\\\" }`. This data is passed to the `assignment_callback_url` when TaskRouter assigns a Task to the Worker. Defaults to {}. */
    attributes?: string;
    /** A descriptive string that you create to describe the Worker. It can be up to 64 characters long. */
    friendlyName?: string;
    /** Whether to reject the Worker\\\'s pending reservations. This option is only valid if the Worker\\\'s new [Activity](https://www.twilio.com/docs/taskrouter/api/activity) resource has its `availability` property set to `False`. */
    rejectPendingReservations?: boolean;
}
/**
 * Options to pass to create a WorkerInstance
 */
export interface WorkerListInstanceCreateOptions {
    /** A descriptive string that you create to describe the new Worker. It can be up to 64 characters long. */
    friendlyName: string;
    /** The SID of a valid Activity that will describe the new Worker\\\'s initial state. See [Activities](https://www.twilio.com/docs/taskrouter/api/activity) for more information. If not provided, the new Worker\\\'s initial state is the `default_activity_sid` configured on the Workspace. */
    activitySid?: string;
    /** A valid JSON string that describes the new Worker. For example: `{ \\\"email\\\": \\\"Bob@example.com\\\", \\\"phone\\\": \\\"+5095551234\\\" }`. This data is passed to the `assignment_callback_url` when TaskRouter assigns a Task to the Worker. Defaults to {}. */
    attributes?: string;
}
/**
 * Options to pass to each
 */
export interface WorkerListInstanceEachOptions {
    /** The `activity_name` of the Worker resources to read. */
    activityName?: string;
    /** The `activity_sid` of the Worker resources to read. */
    activitySid?: string;
    /** Whether to return only Worker resources that are available or unavailable. Can be `true`, `1`, or `yes` to return Worker resources that are available, and `false`, or any value returns the Worker resources that are not available. */
    available?: string;
    /** The `friendly_name` of the Worker resources to read. */
    friendlyName?: string;
    /** Filter by Workers that would match an expression. In addition to fields in the workers\' attributes, the expression can include the following worker fields: `sid`, `friendly_name`, `activity_sid`, or `activity_name` */
    targetWorkersExpression?: string;
    /** The `friendly_name` of the TaskQueue that the Workers to read are eligible for. */
    taskQueueName?: string;
    /** The SID of the TaskQueue that the Workers to read are eligible for. */
    taskQueueSid?: string;
    /** Sorting parameter for Workers */
    ordering?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: WorkerInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface WorkerListInstanceOptions {
    /** The `activity_name` of the Worker resources to read. */
    activityName?: string;
    /** The `activity_sid` of the Worker resources to read. */
    activitySid?: string;
    /** Whether to return only Worker resources that are available or unavailable. Can be `true`, `1`, or `yes` to return Worker resources that are available, and `false`, or any value returns the Worker resources that are not available. */
    available?: string;
    /** The `friendly_name` of the Worker resources to read. */
    friendlyName?: string;
    /** Filter by Workers that would match an expression. In addition to fields in the workers\' attributes, the expression can include the following worker fields: `sid`, `friendly_name`, `activity_sid`, or `activity_name` */
    targetWorkersExpression?: string;
    /** The `friendly_name` of the TaskQueue that the Workers to read are eligible for. */
    taskQueueName?: string;
    /** The SID of the TaskQueue that the Workers to read are eligible for. */
    taskQueueSid?: string;
    /** Sorting parameter for Workers */
    ordering?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface WorkerListInstancePageOptions {
    /** The `activity_name` of the Worker resources to read. */
    activityName?: string;
    /** The `activity_sid` of the Worker resources to read. */
    activitySid?: string;
    /** Whether to return only Worker resources that are available or unavailable. Can be `true`, `1`, or `yes` to return Worker resources that are available, and `false`, or any value returns the Worker resources that are not available. */
    available?: string;
    /** The `friendly_name` of the Worker resources to read. */
    friendlyName?: string;
    /** Filter by Workers that would match an expression. In addition to fields in the workers\' attributes, the expression can include the following worker fields: `sid`, `friendly_name`, `activity_sid`, or `activity_name` */
    targetWorkersExpression?: string;
    /** The `friendly_name` of the TaskQueue that the Workers to read are eligible for. */
    taskQueueName?: string;
    /** The SID of the TaskQueue that the Workers to read are eligible for. */
    taskQueueSid?: string;
    /** Sorting parameter for Workers */
    ordering?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface WorkerContext {
    reservations: ReservationListInstance;
    workerChannels: WorkerChannelListInstance;
    statistics: WorkerStatisticsListInstance;
    /**
     * Remove a WorkerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a WorkerInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerInstance
     */
    remove(params: WorkerContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a WorkerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkerInstance) => any): Promise<WorkerInstance>;
    /**
     * Update a WorkerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerInstance
     */
    update(callback?: (error: Error | null, item?: WorkerInstance) => any): Promise<WorkerInstance>;
    /**
     * Update a WorkerInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerInstance
     */
    update(params: WorkerContextUpdateOptions, callback?: (error: Error | null, item?: WorkerInstance) => any): Promise<WorkerInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WorkerContextSolution {
    workspaceSid: string;
    sid: string;
}
export declare class WorkerContextImpl implements WorkerContext {
    protected _version: V1;
    protected _solution: WorkerContextSolution;
    protected _uri: string;
    protected _reservations?: ReservationListInstance;
    protected _workerChannels?: WorkerChannelListInstance;
    protected _statistics?: WorkerStatisticsListInstance;
    constructor(_version: V1, workspaceSid: string, sid: string);
    get reservations(): ReservationListInstance;
    get workerChannels(): WorkerChannelListInstance;
    get statistics(): WorkerStatisticsListInstance;
    remove(params?: WorkerContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: WorkerInstance) => any): Promise<WorkerInstance>;
    update(params?: WorkerContextUpdateOptions | ((error: Error | null, item?: WorkerInstance) => any), callback?: (error: Error | null, item?: WorkerInstance) => any): Promise<WorkerInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WorkerContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WorkerPayload extends TwilioResponsePayload {
    workers: WorkerResource[];
}
interface WorkerResource {
    account_sid: string;
    activity_name: string;
    activity_sid: string;
    attributes: string;
    available: boolean;
    date_created: Date;
    date_status_changed: Date;
    date_updated: Date;
    friendly_name: string;
    sid: string;
    workspace_sid: string;
    url: string;
    links: Record<string, string>;
}
export declare class WorkerInstance {
    protected _version: V1;
    protected _solution: WorkerContextSolution;
    protected _context?: WorkerContext;
    constructor(_version: V1, payload: WorkerResource, workspaceSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Worker resource.
     */
    accountSid: string;
    /**
     * The `friendly_name` of the Worker\'s current Activity.
     */
    activityName: string;
    /**
     * The SID of the Worker\'s current Activity.
     */
    activitySid: string;
    /**
     * The JSON string that describes the Worker. For example: `{ \"email\": \"Bob@example.com\", \"phone\": \"+5095551234\" }`. **Note** If this property has been assigned a value, it will only be displayed in FETCH actions that return a single resource. Otherwise, this property will be null, even if it has a value. This data is passed to the `assignment_callback_url` when TaskRouter assigns a Task to the Worker.
     */
    attributes: string;
    /**
     * Whether the Worker is available to perform tasks.
     */
    available: boolean;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT of the last change to the Worker\'s activity specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. Used to calculate Workflow statistics.
     */
    dateStatusChanged: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The string that you assigned to describe the resource. Friendly names are case insensitive, and unique within the TaskRouter Workspace.
     */
    friendlyName: string;
    /**
     * The unique string that we created to identify the Worker resource.
     */
    sid: string;
    /**
     * The SID of the Workspace that contains the Worker.
     */
    workspaceSid: string;
    /**
     * The absolute URL of the Worker resource.
     */
    url: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a WorkerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a WorkerInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerInstance
     */
    remove(params: WorkerContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a WorkerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkerInstance) => any): Promise<WorkerInstance>;
    /**
     * Update a WorkerInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerInstance
     */
    update(callback?: (error: Error | null, item?: WorkerInstance) => any): Promise<WorkerInstance>;
    /**
     * Update a WorkerInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerInstance
     */
    update(params: WorkerContextUpdateOptions, callback?: (error: Error | null, item?: WorkerInstance) => any): Promise<WorkerInstance>;
    /**
     * Access the reservations.
     */
    reservations(): ReservationListInstance;
    /**
     * Access the workerChannels.
     */
    workerChannels(): WorkerChannelListInstance;
    /**
     * Access the statistics.
     */
    statistics(): WorkerStatisticsListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        activityName: string;
        activitySid: string;
        attributes: string;
        available: boolean;
        dateCreated: Date;
        dateStatusChanged: Date;
        dateUpdated: Date;
        friendlyName: string;
        sid: string;
        workspaceSid: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WorkerSolution {
    workspaceSid: string;
}
export interface WorkerListInstance {
    _version: V1;
    _solution: WorkerSolution;
    _uri: string;
    (sid: string): WorkerContext;
    get(sid: string): WorkerContext;
    _cumulativeStatistics?: WorkersCumulativeStatisticsListInstance;
    cumulativeStatistics: WorkersCumulativeStatisticsListInstance;
    _realTimeStatistics?: WorkersRealTimeStatisticsListInstance;
    realTimeStatistics: WorkersRealTimeStatisticsListInstance;
    _statistics?: WorkersStatisticsListInstance;
    statistics: WorkersStatisticsListInstance;
    /**
     * Create a WorkerInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerInstance
     */
    create(params: WorkerListInstanceCreateOptions, callback?: (error: Error | null, item?: WorkerInstance) => any): Promise<WorkerInstance>;
    /**
     * Streams WorkerInstance records from the API.
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
     * @param { WorkerListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: WorkerInstance, done: (err?: Error) => void) => void): void;
    each(params: WorkerListInstanceEachOptions, callback?: (item: WorkerInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of WorkerInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: WorkerPage) => any): Promise<WorkerPage>;
    /**
     * Lists WorkerInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { WorkerListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: WorkerInstance[]) => any): Promise<WorkerInstance[]>;
    list(params: WorkerListInstanceOptions, callback?: (error: Error | null, items: WorkerInstance[]) => any): Promise<WorkerInstance[]>;
    /**
     * Retrieve a single page of WorkerInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { WorkerListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: WorkerPage) => any): Promise<WorkerPage>;
    page(params: WorkerListInstancePageOptions, callback?: (error: Error | null, items: WorkerPage) => any): Promise<WorkerPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WorkerListInstance(version: V1, workspaceSid: string): WorkerListInstance;
export declare class WorkerPage extends Page<V1, WorkerPayload, WorkerResource, WorkerInstance> {
    /**
     * Initialize the WorkerPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: WorkerSolution);
    /**
     * Build an instance of WorkerInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: WorkerResource): WorkerInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
