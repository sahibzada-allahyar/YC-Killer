/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
/**
 * Options to pass to fetch a WorkersCumulativeStatisticsInstance
 */
export interface WorkersCumulativeStatisticsContextFetchOptions {
    /** Only calculate statistics from this date and time and earlier, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    endDate?: Date;
    /** Only calculate statistics since this many minutes in the past. The default 15 minutes. This is helpful for displaying statistics for the last 15 minutes, 240 minutes (4 hours), and 480 minutes (8 hours) to see trends. */
    minutes?: number;
    /** Only calculate statistics from this date and time and later, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only calculate cumulative statistics on this TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
}
export interface WorkersCumulativeStatisticsContext {
    /**
     * Fetch a WorkersCumulativeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkersCumulativeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkersCumulativeStatisticsInstance) => any): Promise<WorkersCumulativeStatisticsInstance>;
    /**
     * Fetch a WorkersCumulativeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkersCumulativeStatisticsInstance
     */
    fetch(params: WorkersCumulativeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkersCumulativeStatisticsInstance) => any): Promise<WorkersCumulativeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WorkersCumulativeStatisticsContextSolution {
    workspaceSid: string;
}
export declare class WorkersCumulativeStatisticsContextImpl implements WorkersCumulativeStatisticsContext {
    protected _version: V1;
    protected _solution: WorkersCumulativeStatisticsContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string);
    fetch(params?: WorkersCumulativeStatisticsContextFetchOptions | ((error: Error | null, item?: WorkersCumulativeStatisticsInstance) => any), callback?: (error: Error | null, item?: WorkersCumulativeStatisticsInstance) => any): Promise<WorkersCumulativeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WorkersCumulativeStatisticsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WorkersCumulativeStatisticsResource {
    account_sid: string;
    start_time: Date;
    end_time: Date;
    activity_durations: Array<any>;
    reservations_created: number;
    reservations_accepted: number;
    reservations_rejected: number;
    reservations_timed_out: number;
    reservations_canceled: number;
    reservations_rescinded: number;
    workspace_sid: string;
    url: string;
}
export declare class WorkersCumulativeStatisticsInstance {
    protected _version: V1;
    protected _solution: WorkersCumulativeStatisticsContextSolution;
    protected _context?: WorkersCumulativeStatisticsContext;
    constructor(_version: V1, payload: WorkersCumulativeStatisticsResource, workspaceSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Worker resource.
     */
    accountSid: string;
    /**
     * The beginning of the interval during which these statistics were calculated, in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    startTime: Date;
    /**
     * The end of the interval during which these statistics were calculated, in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    endTime: Date;
    /**
     * The minimum, average, maximum, and total time (in seconds) that Workers spent in each Activity.
     */
    activityDurations: Array<any>;
    /**
     * The total number of Reservations that were created.
     */
    reservationsCreated: number;
    /**
     * The total number of Reservations that were accepted.
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
     * The SID of the Workspace that contains the Workers.
     */
    workspaceSid: string;
    /**
     * The absolute URL of the Workers statistics resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a WorkersCumulativeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkersCumulativeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkersCumulativeStatisticsInstance) => any): Promise<WorkersCumulativeStatisticsInstance>;
    /**
     * Fetch a WorkersCumulativeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkersCumulativeStatisticsInstance
     */
    fetch(params: WorkersCumulativeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkersCumulativeStatisticsInstance) => any): Promise<WorkersCumulativeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        startTime: Date;
        endTime: Date;
        activityDurations: any[];
        reservationsCreated: number;
        reservationsAccepted: number;
        reservationsRejected: number;
        reservationsTimedOut: number;
        reservationsCanceled: number;
        reservationsRescinded: number;
        workspaceSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WorkersCumulativeStatisticsSolution {
    workspaceSid: string;
}
export interface WorkersCumulativeStatisticsListInstance {
    _version: V1;
    _solution: WorkersCumulativeStatisticsSolution;
    _uri: string;
    (): WorkersCumulativeStatisticsContext;
    get(): WorkersCumulativeStatisticsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WorkersCumulativeStatisticsListInstance(version: V1, workspaceSid: string): WorkersCumulativeStatisticsListInstance;
export {};
