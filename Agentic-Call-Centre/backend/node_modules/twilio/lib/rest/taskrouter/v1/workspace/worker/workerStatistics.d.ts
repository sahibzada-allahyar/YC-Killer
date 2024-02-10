/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
/**
 * Options to pass to fetch a WorkerStatisticsInstance
 */
export interface WorkerStatisticsContextFetchOptions {
    /** Only calculate statistics since this many minutes in the past. The default 15 minutes. This is helpful for displaying statistics for the last 15 minutes, 240 minutes (4 hours), and 480 minutes (8 hours) to see trends. */
    minutes?: number;
    /** Only calculate statistics from this date and time and later, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only include usage that occurred on or before this date, specified in GMT as an [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time. */
    endDate?: Date;
    /** Only calculate statistics on this TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
}
export interface WorkerStatisticsContext {
    /**
     * Fetch a WorkerStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkerStatisticsInstance) => any): Promise<WorkerStatisticsInstance>;
    /**
     * Fetch a WorkerStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerStatisticsInstance
     */
    fetch(params: WorkerStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkerStatisticsInstance) => any): Promise<WorkerStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WorkerStatisticsContextSolution {
    workspaceSid: string;
    workerSid: string;
}
export declare class WorkerStatisticsContextImpl implements WorkerStatisticsContext {
    protected _version: V1;
    protected _solution: WorkerStatisticsContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string, workerSid: string);
    fetch(params?: WorkerStatisticsContextFetchOptions | ((error: Error | null, item?: WorkerStatisticsInstance) => any), callback?: (error: Error | null, item?: WorkerStatisticsInstance) => any): Promise<WorkerStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WorkerStatisticsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WorkerStatisticsResource {
    account_sid: string;
    cumulative: any;
    worker_sid: string;
    workspace_sid: string;
    url: string;
}
export declare class WorkerStatisticsInstance {
    protected _version: V1;
    protected _solution: WorkerStatisticsContextSolution;
    protected _context?: WorkerStatisticsContext;
    constructor(_version: V1, payload: WorkerStatisticsResource, workspaceSid: string, workerSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Worker resource.
     */
    accountSid: string;
    /**
     * An object that contains the cumulative statistics for the Worker.
     */
    cumulative: any;
    /**
     * The SID of the Worker that contains the WorkerChannel.
     */
    workerSid: string;
    /**
     * The SID of the Workspace that contains the WorkerChannel.
     */
    workspaceSid: string;
    /**
     * The absolute URL of the WorkerChannel statistics resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a WorkerStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkerStatisticsInstance) => any): Promise<WorkerStatisticsInstance>;
    /**
     * Fetch a WorkerStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkerStatisticsInstance
     */
    fetch(params: WorkerStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkerStatisticsInstance) => any): Promise<WorkerStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        cumulative: any;
        workerSid: string;
        workspaceSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WorkerStatisticsSolution {
    workspaceSid: string;
    workerSid: string;
}
export interface WorkerStatisticsListInstance {
    _version: V1;
    _solution: WorkerStatisticsSolution;
    _uri: string;
    (): WorkerStatisticsContext;
    get(): WorkerStatisticsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WorkerStatisticsListInstance(version: V1, workspaceSid: string, workerSid: string): WorkerStatisticsListInstance;
export {};
