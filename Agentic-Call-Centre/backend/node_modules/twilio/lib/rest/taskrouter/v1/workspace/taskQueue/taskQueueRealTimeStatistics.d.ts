/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
/**
 * Options to pass to fetch a TaskQueueRealTimeStatisticsInstance
 */
export interface TaskQueueRealTimeStatisticsContextFetchOptions {
    /** The TaskChannel for which to fetch statistics. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
}
export interface TaskQueueRealTimeStatisticsContext {
    /**
     * Fetch a TaskQueueRealTimeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueRealTimeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: TaskQueueRealTimeStatisticsInstance) => any): Promise<TaskQueueRealTimeStatisticsInstance>;
    /**
     * Fetch a TaskQueueRealTimeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueRealTimeStatisticsInstance
     */
    fetch(params: TaskQueueRealTimeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: TaskQueueRealTimeStatisticsInstance) => any): Promise<TaskQueueRealTimeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TaskQueueRealTimeStatisticsContextSolution {
    workspaceSid: string;
    taskQueueSid: string;
}
export declare class TaskQueueRealTimeStatisticsContextImpl implements TaskQueueRealTimeStatisticsContext {
    protected _version: V1;
    protected _solution: TaskQueueRealTimeStatisticsContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string, taskQueueSid: string);
    fetch(params?: TaskQueueRealTimeStatisticsContextFetchOptions | ((error: Error | null, item?: TaskQueueRealTimeStatisticsInstance) => any), callback?: (error: Error | null, item?: TaskQueueRealTimeStatisticsInstance) => any): Promise<TaskQueueRealTimeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TaskQueueRealTimeStatisticsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TaskQueueRealTimeStatisticsResource {
    account_sid: string;
    activity_statistics: Array<any>;
    longest_task_waiting_age: number;
    longest_task_waiting_sid: string;
    longest_relative_task_age_in_queue: number;
    longest_relative_task_sid_in_queue: string;
    task_queue_sid: string;
    tasks_by_priority: any;
    tasks_by_status: any;
    total_available_workers: number;
    total_eligible_workers: number;
    total_tasks: number;
    workspace_sid: string;
    url: string;
}
export declare class TaskQueueRealTimeStatisticsInstance {
    protected _version: V1;
    protected _solution: TaskQueueRealTimeStatisticsContextSolution;
    protected _context?: TaskQueueRealTimeStatisticsContext;
    constructor(_version: V1, payload: TaskQueueRealTimeStatisticsResource, workspaceSid: string, taskQueueSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the TaskQueue resource.
     */
    accountSid: string;
    /**
     * The number of current Workers by Activity.
     */
    activityStatistics: Array<any>;
    /**
     * The age of the longest waiting Task.
     */
    longestTaskWaitingAge: number;
    /**
     * The SID of the longest waiting Task.
     */
    longestTaskWaitingSid: string;
    /**
     * The relative age in the TaskQueue for the longest waiting Task. Calculation is based on the time when the Task entered the TaskQueue.
     */
    longestRelativeTaskAgeInQueue: number;
    /**
     * The Task SID of the Task waiting in the TaskQueue the longest. Calculation is based on the time when the Task entered the TaskQueue.
     */
    longestRelativeTaskSidInQueue: string;
    /**
     * The SID of the TaskQueue from which these statistics were calculated.
     */
    taskQueueSid: string;
    /**
     * The number of Tasks by priority. For example: `{\"0\": \"10\", \"99\": \"5\"}` shows 10 Tasks at priority 0 and 5 at priority 99.
     */
    tasksByPriority: any;
    /**
     * The number of Tasks by their current status. For example: `{\"pending\": \"1\", \"reserved\": \"3\", \"assigned\": \"2\", \"completed\": \"5\"}`.
     */
    tasksByStatus: any;
    /**
     * The total number of Workers available for Tasks in the TaskQueue.
     */
    totalAvailableWorkers: number;
    /**
     * The total number of Workers eligible for Tasks in the TaskQueue, independent of their Activity state.
     */
    totalEligibleWorkers: number;
    /**
     * The total number of Tasks.
     */
    totalTasks: number;
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
     * Fetch a TaskQueueRealTimeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueRealTimeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: TaskQueueRealTimeStatisticsInstance) => any): Promise<TaskQueueRealTimeStatisticsInstance>;
    /**
     * Fetch a TaskQueueRealTimeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueRealTimeStatisticsInstance
     */
    fetch(params: TaskQueueRealTimeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: TaskQueueRealTimeStatisticsInstance) => any): Promise<TaskQueueRealTimeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        activityStatistics: any[];
        longestTaskWaitingAge: number;
        longestTaskWaitingSid: string;
        longestRelativeTaskAgeInQueue: number;
        longestRelativeTaskSidInQueue: string;
        taskQueueSid: string;
        tasksByPriority: any;
        tasksByStatus: any;
        totalAvailableWorkers: number;
        totalEligibleWorkers: number;
        totalTasks: number;
        workspaceSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TaskQueueRealTimeStatisticsSolution {
    workspaceSid: string;
    taskQueueSid: string;
}
export interface TaskQueueRealTimeStatisticsListInstance {
    _version: V1;
    _solution: TaskQueueRealTimeStatisticsSolution;
    _uri: string;
    (): TaskQueueRealTimeStatisticsContext;
    get(): TaskQueueRealTimeStatisticsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TaskQueueRealTimeStatisticsListInstance(version: V1, workspaceSid: string, taskQueueSid: string): TaskQueueRealTimeStatisticsListInstance;
export {};
