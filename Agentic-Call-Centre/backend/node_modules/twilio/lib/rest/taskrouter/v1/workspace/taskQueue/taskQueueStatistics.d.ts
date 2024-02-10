/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
/**
 * Options to pass to fetch a TaskQueueStatisticsInstance
 */
export interface TaskQueueStatisticsContextFetchOptions {
    /** Only calculate statistics from this date and time and earlier, specified in GMT as an [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time. */
    endDate?: Date;
    /** Only calculate statistics since this many minutes in the past. The default is 15 minutes. */
    minutes?: number;
    /** Only calculate statistics from this date and time and later, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only calculate real-time and cumulative statistics for the specified TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
    /** A comma separated list of values that describes the thresholds, in seconds, to calculate statistics on. For each threshold specified, the number of Tasks canceled and reservations accepted above and below the specified thresholds in seconds are computed. */
    splitByWaitTime?: string;
}
export interface TaskQueueStatisticsContext {
    /**
     * Fetch a TaskQueueStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: TaskQueueStatisticsInstance) => any): Promise<TaskQueueStatisticsInstance>;
    /**
     * Fetch a TaskQueueStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueStatisticsInstance
     */
    fetch(params: TaskQueueStatisticsContextFetchOptions, callback?: (error: Error | null, item?: TaskQueueStatisticsInstance) => any): Promise<TaskQueueStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TaskQueueStatisticsContextSolution {
    workspaceSid: string;
    taskQueueSid: string;
}
export declare class TaskQueueStatisticsContextImpl implements TaskQueueStatisticsContext {
    protected _version: V1;
    protected _solution: TaskQueueStatisticsContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string, taskQueueSid: string);
    fetch(params?: TaskQueueStatisticsContextFetchOptions | ((error: Error | null, item?: TaskQueueStatisticsInstance) => any), callback?: (error: Error | null, item?: TaskQueueStatisticsInstance) => any): Promise<TaskQueueStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TaskQueueStatisticsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TaskQueueStatisticsResource {
    account_sid: string;
    cumulative: any;
    realtime: any;
    task_queue_sid: string;
    workspace_sid: string;
    url: string;
}
export declare class TaskQueueStatisticsInstance {
    protected _version: V1;
    protected _solution: TaskQueueStatisticsContextSolution;
    protected _context?: TaskQueueStatisticsContext;
    constructor(_version: V1, payload: TaskQueueStatisticsResource, workspaceSid: string, taskQueueSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the TaskQueue resource.
     */
    accountSid: string;
    /**
     * An object that contains the cumulative statistics for the TaskQueue.
     */
    cumulative: any;
    /**
     * An object that contains the real-time statistics for the TaskQueue.
     */
    realtime: any;
    /**
     * The SID of the TaskQueue from which these statistics were calculated.
     */
    taskQueueSid: string;
    /**
     * The SID of the Workspace that contains the TaskQueue.
     */
    workspaceSid: string;
    /**
     * The absolute URL of the TaskQueue statistics resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a TaskQueueStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: TaskQueueStatisticsInstance) => any): Promise<TaskQueueStatisticsInstance>;
    /**
     * Fetch a TaskQueueStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueStatisticsInstance
     */
    fetch(params: TaskQueueStatisticsContextFetchOptions, callback?: (error: Error | null, item?: TaskQueueStatisticsInstance) => any): Promise<TaskQueueStatisticsInstance>;
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
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TaskQueueStatisticsSolution {
    workspaceSid: string;
    taskQueueSid: string;
}
export interface TaskQueueStatisticsListInstance {
    _version: V1;
    _solution: TaskQueueStatisticsSolution;
    _uri: string;
    (): TaskQueueStatisticsContext;
    get(): TaskQueueStatisticsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TaskQueueStatisticsListInstance(version: V1, workspaceSid: string, taskQueueSid: string): TaskQueueStatisticsListInstance;
export {};
