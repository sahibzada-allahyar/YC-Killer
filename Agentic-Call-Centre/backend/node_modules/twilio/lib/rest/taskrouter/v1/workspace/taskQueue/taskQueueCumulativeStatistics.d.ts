/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
/**
 * Options to pass to fetch a TaskQueueCumulativeStatisticsInstance
 */
export interface TaskQueueCumulativeStatisticsContextFetchOptions {
    /** Only calculate statistics from this date and time and earlier, specified in GMT as an [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time. */
    endDate?: Date;
    /** Only calculate statistics since this many minutes in the past. The default is 15 minutes. */
    minutes?: number;
    /** Only calculate statistics from this date and time and later, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only calculate cumulative statistics on this TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
    /** A comma separated list of values that describes the thresholds, in seconds, to calculate statistics on. For each threshold specified, the number of Tasks canceled and reservations accepted above and below the specified thresholds in seconds are computed. TaskRouter will calculate statistics on up to 10,000 Tasks/Reservations for any given threshold. */
    splitByWaitTime?: string;
}
export interface TaskQueueCumulativeStatisticsContext {
    /**
     * Fetch a TaskQueueCumulativeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueCumulativeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: TaskQueueCumulativeStatisticsInstance) => any): Promise<TaskQueueCumulativeStatisticsInstance>;
    /**
     * Fetch a TaskQueueCumulativeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueCumulativeStatisticsInstance
     */
    fetch(params: TaskQueueCumulativeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: TaskQueueCumulativeStatisticsInstance) => any): Promise<TaskQueueCumulativeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TaskQueueCumulativeStatisticsContextSolution {
    workspaceSid: string;
    taskQueueSid: string;
}
export declare class TaskQueueCumulativeStatisticsContextImpl implements TaskQueueCumulativeStatisticsContext {
    protected _version: V1;
    protected _solution: TaskQueueCumulativeStatisticsContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string, taskQueueSid: string);
    fetch(params?: TaskQueueCumulativeStatisticsContextFetchOptions | ((error: Error | null, item?: TaskQueueCumulativeStatisticsInstance) => any), callback?: (error: Error | null, item?: TaskQueueCumulativeStatisticsInstance) => any): Promise<TaskQueueCumulativeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TaskQueueCumulativeStatisticsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TaskQueueCumulativeStatisticsResource {
    account_sid: string;
    avg_task_acceptance_time: number;
    start_time: Date;
    end_time: Date;
    reservations_created: number;
    reservations_accepted: number;
    reservations_rejected: number;
    reservations_timed_out: number;
    reservations_canceled: number;
    reservations_rescinded: number;
    split_by_wait_time: any;
    task_queue_sid: string;
    wait_duration_until_accepted: any;
    wait_duration_until_canceled: any;
    wait_duration_in_queue_until_accepted: any;
    tasks_canceled: number;
    tasks_completed: number;
    tasks_deleted: number;
    tasks_entered: number;
    tasks_moved: number;
    workspace_sid: string;
    url: string;
}
export declare class TaskQueueCumulativeStatisticsInstance {
    protected _version: V1;
    protected _solution: TaskQueueCumulativeStatisticsContextSolution;
    protected _context?: TaskQueueCumulativeStatisticsContext;
    constructor(_version: V1, payload: TaskQueueCumulativeStatisticsResource, workspaceSid: string, taskQueueSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the TaskQueue resource.
     */
    accountSid: string;
    /**
     * The average time in seconds between Task creation and acceptance.
     */
    avgTaskAcceptanceTime: number;
    /**
     * The beginning of the interval during which these statistics were calculated, in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    startTime: Date;
    /**
     * The end of the interval during which these statistics were calculated, in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    endTime: Date;
    /**
     * The total number of Reservations created for Tasks in the TaskQueue.
     */
    reservationsCreated: number;
    /**
     * The total number of Reservations accepted for Tasks in the TaskQueue.
     */
    reservationsAccepted: number;
    /**
     * The total number of Reservations rejected for Tasks in the TaskQueue.
     */
    reservationsRejected: number;
    /**
     * The total number of Reservations that timed out for Tasks in the TaskQueue.
     */
    reservationsTimedOut: number;
    /**
     * The total number of Reservations canceled for Tasks in the TaskQueue.
     */
    reservationsCanceled: number;
    /**
     * The total number of Reservations rescinded.
     */
    reservationsRescinded: number;
    /**
     * A list of objects that describe the number of Tasks canceled and reservations accepted above and below the thresholds specified in seconds.
     */
    splitByWaitTime: any;
    /**
     * The SID of the TaskQueue from which these statistics were calculated.
     */
    taskQueueSid: string;
    /**
     * The wait duration statistics (`avg`, `min`, `max`, `total`) for Tasks accepted while in the TaskQueue. Calculation is based on the time when the Tasks were created. For transfers, the wait duration is counted from the moment ***the Task was created***, and not from when the transfer was initiated.
     */
    waitDurationUntilAccepted: any;
    /**
     * The wait duration statistics (`avg`, `min`, `max`, `total`) for Tasks canceled while in the TaskQueue.
     */
    waitDurationUntilCanceled: any;
    /**
     * The relative wait duration statistics (`avg`, `min`, `max`, `total`) for Tasks accepted while in the TaskQueue. Calculation is based on the time when the Tasks entered the TaskQueue.
     */
    waitDurationInQueueUntilAccepted: any;
    /**
     * The total number of Tasks canceled in the TaskQueue.
     */
    tasksCanceled: number;
    /**
     * The total number of Tasks completed in the TaskQueue.
     */
    tasksCompleted: number;
    /**
     * The total number of Tasks deleted in the TaskQueue.
     */
    tasksDeleted: number;
    /**
     * The total number of Tasks entered into the TaskQueue.
     */
    tasksEntered: number;
    /**
     * The total number of Tasks that were moved from one queue to another.
     */
    tasksMoved: number;
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
     * Fetch a TaskQueueCumulativeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueCumulativeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: TaskQueueCumulativeStatisticsInstance) => any): Promise<TaskQueueCumulativeStatisticsInstance>;
    /**
     * Fetch a TaskQueueCumulativeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TaskQueueCumulativeStatisticsInstance
     */
    fetch(params: TaskQueueCumulativeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: TaskQueueCumulativeStatisticsInstance) => any): Promise<TaskQueueCumulativeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        avgTaskAcceptanceTime: number;
        startTime: Date;
        endTime: Date;
        reservationsCreated: number;
        reservationsAccepted: number;
        reservationsRejected: number;
        reservationsTimedOut: number;
        reservationsCanceled: number;
        reservationsRescinded: number;
        splitByWaitTime: any;
        taskQueueSid: string;
        waitDurationUntilAccepted: any;
        waitDurationUntilCanceled: any;
        waitDurationInQueueUntilAccepted: any;
        tasksCanceled: number;
        tasksCompleted: number;
        tasksDeleted: number;
        tasksEntered: number;
        tasksMoved: number;
        workspaceSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface TaskQueueCumulativeStatisticsSolution {
    workspaceSid: string;
    taskQueueSid: string;
}
export interface TaskQueueCumulativeStatisticsListInstance {
    _version: V1;
    _solution: TaskQueueCumulativeStatisticsSolution;
    _uri: string;
    (): TaskQueueCumulativeStatisticsContext;
    get(): TaskQueueCumulativeStatisticsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TaskQueueCumulativeStatisticsListInstance(version: V1, workspaceSid: string, taskQueueSid: string): TaskQueueCumulativeStatisticsListInstance;
export {};
