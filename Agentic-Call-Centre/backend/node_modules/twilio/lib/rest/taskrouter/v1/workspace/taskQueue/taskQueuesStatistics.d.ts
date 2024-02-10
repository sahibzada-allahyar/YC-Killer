/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
/**
 * Options to pass to each
 */
export interface TaskQueuesStatisticsListInstanceEachOptions {
    /** Only calculate statistics from this date and time and earlier, specified in GMT as an [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time. */
    endDate?: Date;
    /** The `friendly_name` of the TaskQueue statistics to read. */
    friendlyName?: string;
    /** Only calculate statistics since this many minutes in the past. The default is 15 minutes. */
    minutes?: number;
    /** Only calculate statistics from this date and time and later, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only calculate statistics on this TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
    /** A comma separated list of values that describes the thresholds, in seconds, to calculate statistics on. For each threshold specified, the number of Tasks canceled and reservations accepted above and below the specified thresholds in seconds are computed. */
    splitByWaitTime?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TaskQueuesStatisticsInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TaskQueuesStatisticsListInstanceOptions {
    /** Only calculate statistics from this date and time and earlier, specified in GMT as an [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time. */
    endDate?: Date;
    /** The `friendly_name` of the TaskQueue statistics to read. */
    friendlyName?: string;
    /** Only calculate statistics since this many minutes in the past. The default is 15 minutes. */
    minutes?: number;
    /** Only calculate statistics from this date and time and later, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only calculate statistics on this TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
    /** A comma separated list of values that describes the thresholds, in seconds, to calculate statistics on. For each threshold specified, the number of Tasks canceled and reservations accepted above and below the specified thresholds in seconds are computed. */
    splitByWaitTime?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TaskQueuesStatisticsListInstancePageOptions {
    /** Only calculate statistics from this date and time and earlier, specified in GMT as an [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time. */
    endDate?: Date;
    /** The `friendly_name` of the TaskQueue statistics to read. */
    friendlyName?: string;
    /** Only calculate statistics since this many minutes in the past. The default is 15 minutes. */
    minutes?: number;
    /** Only calculate statistics from this date and time and later, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only calculate statistics on this TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
    /** A comma separated list of values that describes the thresholds, in seconds, to calculate statistics on. For each threshold specified, the number of Tasks canceled and reservations accepted above and below the specified thresholds in seconds are computed. */
    splitByWaitTime?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TaskQueuesStatisticsSolution {
    workspaceSid: string;
}
export interface TaskQueuesStatisticsListInstance {
    _version: V1;
    _solution: TaskQueuesStatisticsSolution;
    _uri: string;
    /**
     * Streams TaskQueuesStatisticsInstance records from the API.
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
     * @param { TaskQueuesStatisticsListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TaskQueuesStatisticsInstance, done: (err?: Error) => void) => void): void;
    each(params: TaskQueuesStatisticsListInstanceEachOptions, callback?: (item: TaskQueuesStatisticsInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TaskQueuesStatisticsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TaskQueuesStatisticsPage) => any): Promise<TaskQueuesStatisticsPage>;
    /**
     * Lists TaskQueuesStatisticsInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TaskQueuesStatisticsListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TaskQueuesStatisticsInstance[]) => any): Promise<TaskQueuesStatisticsInstance[]>;
    list(params: TaskQueuesStatisticsListInstanceOptions, callback?: (error: Error | null, items: TaskQueuesStatisticsInstance[]) => any): Promise<TaskQueuesStatisticsInstance[]>;
    /**
     * Retrieve a single page of TaskQueuesStatisticsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TaskQueuesStatisticsListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TaskQueuesStatisticsPage) => any): Promise<TaskQueuesStatisticsPage>;
    page(params: TaskQueuesStatisticsListInstancePageOptions, callback?: (error: Error | null, items: TaskQueuesStatisticsPage) => any): Promise<TaskQueuesStatisticsPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TaskQueuesStatisticsListInstance(version: V1, workspaceSid: string): TaskQueuesStatisticsListInstance;
interface TaskQueuesStatisticsPayload extends TwilioResponsePayload {
    task_queues_statistics: TaskQueuesStatisticsResource[];
}
interface TaskQueuesStatisticsResource {
    account_sid: string;
    cumulative: any;
    realtime: any;
    task_queue_sid: string;
    workspace_sid: string;
}
export declare class TaskQueuesStatisticsInstance {
    protected _version: V1;
    constructor(_version: V1, payload: TaskQueuesStatisticsResource, workspaceSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the TaskQueue resource.
     */
    accountSid: string;
    /**
     * An object that contains the cumulative statistics for the TaskQueues.
     */
    cumulative: any;
    /**
     * An object that contains the real-time statistics for the TaskQueues.
     */
    realtime: any;
    /**
     * The SID of the TaskQueue from which these statistics were calculated.
     */
    taskQueueSid: string;
    /**
     * The SID of the Workspace that contains the TaskQueues.
     */
    workspaceSid: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        cumulative: any;
        realtime: any;
        taskQueueSid: string;
        workspaceSid: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class TaskQueuesStatisticsPage extends Page<V1, TaskQueuesStatisticsPayload, TaskQueuesStatisticsResource, TaskQueuesStatisticsInstance> {
    /**
     * Initialize the TaskQueuesStatisticsPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: TaskQueuesStatisticsSolution);
    /**
     * Build an instance of TaskQueuesStatisticsInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TaskQueuesStatisticsResource): TaskQueuesStatisticsInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
