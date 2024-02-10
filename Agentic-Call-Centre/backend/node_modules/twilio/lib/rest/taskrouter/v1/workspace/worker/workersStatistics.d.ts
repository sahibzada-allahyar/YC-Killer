/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
/**
 * Options to pass to fetch a WorkersStatisticsInstance
 */
export interface WorkersStatisticsContextFetchOptions {
    /** Only calculate statistics since this many minutes in the past. The default 15 minutes. This is helpful for displaying statistics for the last 15 minutes, 240 minutes (4 hours), and 480 minutes (8 hours) to see trends. */
    minutes?: number;
    /** Only calculate statistics from this date and time and later, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only calculate statistics from this date and time and earlier, specified in GMT as an [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time. */
    endDate?: Date;
    /** The SID of the TaskQueue for which to fetch Worker statistics. */
    taskQueueSid?: string;
    /** The `friendly_name` of the TaskQueue for which to fetch Worker statistics. */
    taskQueueName?: string;
    /** Only include Workers with `friendly_name` values that match this parameter. */
    friendlyName?: string;
    /** Only calculate statistics on this TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
}
export interface WorkersStatisticsContext {
    /**
     * Fetch a WorkersStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkersStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkersStatisticsInstance) => any): Promise<WorkersStatisticsInstance>;
    /**
     * Fetch a WorkersStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkersStatisticsInstance
     */
    fetch(params: WorkersStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkersStatisticsInstance) => any): Promise<WorkersStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WorkersStatisticsContextSolution {
    workspaceSid: string;
}
export declare class WorkersStatisticsContextImpl implements WorkersStatisticsContext {
    protected _version: V1;
    protected _solution: WorkersStatisticsContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string);
    fetch(params?: WorkersStatisticsContextFetchOptions | ((error: Error | null, item?: WorkersStatisticsInstance) => any), callback?: (error: Error | null, item?: WorkersStatisticsInstance) => any): Promise<WorkersStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WorkersStatisticsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WorkersStatisticsResource {
    realtime: any;
    cumulative: any;
    account_sid: string;
    workspace_sid: string;
    url: string;
}
export declare class WorkersStatisticsInstance {
    protected _version: V1;
    protected _solution: WorkersStatisticsContextSolution;
    protected _context?: WorkersStatisticsContext;
    constructor(_version: V1, payload: WorkersStatisticsResource, workspaceSid: string);
    /**
     * An object that contains the real-time statistics for the Worker.
     */
    realtime: any;
    /**
     * An object that contains the cumulative statistics for the Worker.
     */
    cumulative: any;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Worker resource.
     */
    accountSid: string;
    /**
     * The SID of the Workspace that contains the Worker.
     */
    workspaceSid: string;
    /**
     * The absolute URL of the Worker statistics resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a WorkersStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkersStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkersStatisticsInstance) => any): Promise<WorkersStatisticsInstance>;
    /**
     * Fetch a WorkersStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkersStatisticsInstance
     */
    fetch(params: WorkersStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkersStatisticsInstance) => any): Promise<WorkersStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        realtime: any;
        cumulative: any;
        accountSid: string;
        workspaceSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WorkersStatisticsSolution {
    workspaceSid: string;
}
export interface WorkersStatisticsListInstance {
    _version: V1;
    _solution: WorkersStatisticsSolution;
    _uri: string;
    (): WorkersStatisticsContext;
    get(): WorkersStatisticsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WorkersStatisticsListInstance(version: V1, workspaceSid: string): WorkersStatisticsListInstance;
export {};
