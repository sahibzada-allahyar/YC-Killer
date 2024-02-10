/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
/**
 * Options to pass to fetch a WorkflowCumulativeStatisticsInstance
 */
export interface WorkflowCumulativeStatisticsContextFetchOptions {
    /** Only include usage that occurred on or before this date, specified in GMT as an [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time. */
    endDate?: Date;
    /** Only calculate statistics since this many minutes in the past. The default 15 minutes. This is helpful for displaying statistics for the last 15 minutes, 240 minutes (4 hours), and 480 minutes (8 hours) to see trends. */
    minutes?: number;
    /** Only calculate statistics from this date and time and later, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only calculate cumulative statistics on this TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
    /** A comma separated list of values that describes the thresholds, in seconds, to calculate statistics on. For each threshold specified, the number of Tasks canceled and reservations accepted above and below the specified thresholds in seconds are computed. For example, `5,30` would show splits of Tasks that were canceled or accepted before and after 5 seconds and before and after 30 seconds. This can be used to show short abandoned Tasks or Tasks that failed to meet an SLA. TaskRouter will calculate statistics on up to 10,000 Tasks for any given threshold. */
    splitByWaitTime?: string;
}
export interface WorkflowCumulativeStatisticsContext {
    /**
     * Fetch a WorkflowCumulativeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowCumulativeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkflowCumulativeStatisticsInstance) => any): Promise<WorkflowCumulativeStatisticsInstance>;
    /**
     * Fetch a WorkflowCumulativeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowCumulativeStatisticsInstance
     */
    fetch(params: WorkflowCumulativeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkflowCumulativeStatisticsInstance) => any): Promise<WorkflowCumulativeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WorkflowCumulativeStatisticsContextSolution {
    workspaceSid: string;
    workflowSid: string;
}
export declare class WorkflowCumulativeStatisticsContextImpl implements WorkflowCumulativeStatisticsContext {
    protected _version: V1;
    protected _solution: WorkflowCumulativeStatisticsContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string, workflowSid: string);
    fetch(params?: WorkflowCumulativeStatisticsContextFetchOptions | ((error: Error | null, item?: WorkflowCumulativeStatisticsInstance) => any), callback?: (error: Error | null, item?: WorkflowCumulativeStatisticsInstance) => any): Promise<WorkflowCumulativeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WorkflowCumulativeStatisticsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WorkflowCumulativeStatisticsResource {
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
    wait_duration_until_accepted: any;
    wait_duration_until_canceled: any;
    tasks_canceled: number;
    tasks_completed: number;
    tasks_entered: number;
    tasks_deleted: number;
    tasks_moved: number;
    tasks_timed_out_in_workflow: number;
    workflow_sid: string;
    workspace_sid: string;
    url: string;
}
export declare class WorkflowCumulativeStatisticsInstance {
    protected _version: V1;
    protected _solution: WorkflowCumulativeStatisticsContextSolution;
    protected _context?: WorkflowCumulativeStatisticsContext;
    constructor(_version: V1, payload: WorkflowCumulativeStatisticsResource, workspaceSid: string, workflowSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Workflow resource.
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
     * The total number of Reservations that were created for Workers.
     */
    reservationsCreated: number;
    /**
     * The total number of Reservations accepted by Workers.
     */
    reservationsAccepted: number;
    /**
     * The total number of Reservations that were rejected.
     */
    reservationsRejected: number;
    /**
     * The total number of Reservations that were timed out.
     */
    reservationsTimedOut: number;
    /**
     * The total number of Reservations that were canceled.
     */
    reservationsCanceled: number;
    /**
     * The total number of Reservations that were rescinded.
     */
    reservationsRescinded: number;
    /**
     * A list of objects that describe the number of Tasks canceled and reservations accepted above and below the thresholds specified in seconds.
     */
    splitByWaitTime: any;
    /**
     * The wait duration statistics (`avg`, `min`, `max`, `total`) for Tasks that were accepted.
     */
    waitDurationUntilAccepted: any;
    /**
     * The wait duration statistics (`avg`, `min`, `max`, `total`) for Tasks that were canceled.
     */
    waitDurationUntilCanceled: any;
    /**
     * The total number of Tasks that were canceled.
     */
    tasksCanceled: number;
    /**
     * The total number of Tasks that were completed.
     */
    tasksCompleted: number;
    /**
     * The total number of Tasks that entered the Workflow.
     */
    tasksEntered: number;
    /**
     * The total number of Tasks that were deleted.
     */
    tasksDeleted: number;
    /**
     * The total number of Tasks that were moved from one queue to another.
     */
    tasksMoved: number;
    /**
     * The total number of Tasks that were timed out of their Workflows (and deleted).
     */
    tasksTimedOutInWorkflow: number;
    /**
     * Returns the list of Tasks that are being controlled by the Workflow with the specified Sid value.
     */
    workflowSid: string;
    /**
     * The SID of the Workspace that contains the Workflow.
     */
    workspaceSid: string;
    /**
     * The absolute URL of the Workflow statistics resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a WorkflowCumulativeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowCumulativeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkflowCumulativeStatisticsInstance) => any): Promise<WorkflowCumulativeStatisticsInstance>;
    /**
     * Fetch a WorkflowCumulativeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowCumulativeStatisticsInstance
     */
    fetch(params: WorkflowCumulativeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkflowCumulativeStatisticsInstance) => any): Promise<WorkflowCumulativeStatisticsInstance>;
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
        waitDurationUntilAccepted: any;
        waitDurationUntilCanceled: any;
        tasksCanceled: number;
        tasksCompleted: number;
        tasksEntered: number;
        tasksDeleted: number;
        tasksMoved: number;
        tasksTimedOutInWorkflow: number;
        workflowSid: string;
        workspaceSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WorkflowCumulativeStatisticsSolution {
    workspaceSid: string;
    workflowSid: string;
}
export interface WorkflowCumulativeStatisticsListInstance {
    _version: V1;
    _solution: WorkflowCumulativeStatisticsSolution;
    _uri: string;
    (): WorkflowCumulativeStatisticsContext;
    get(): WorkflowCumulativeStatisticsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WorkflowCumulativeStatisticsListInstance(version: V1, workspaceSid: string, workflowSid: string): WorkflowCumulativeStatisticsListInstance;
export {};
