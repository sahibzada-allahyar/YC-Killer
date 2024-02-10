/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
/**
 * Options to pass to fetch a WorkspaceStatisticsInstance
 */
export interface WorkspaceStatisticsContextFetchOptions {
    /** Only calculate statistics since this many minutes in the past. The default 15 minutes. This is helpful for displaying statistics for the last 15 minutes, 240 minutes (4 hours), and 480 minutes (8 hours) to see trends. */
    minutes?: number;
    /** Only calculate statistics from this date and time and later, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only calculate statistics from this date and time and earlier, specified in GMT as an [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time. */
    endDate?: Date;
    /** Only calculate statistics on this TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
    /** A comma separated list of values that describes the thresholds, in seconds, to calculate statistics on. For each threshold specified, the number of Tasks canceled and reservations accepted above and below the specified thresholds in seconds are computed. For example, `5,30` would show splits of Tasks that were canceled or accepted before and after 5 seconds and before and after 30 seconds. This can be used to show short abandoned Tasks or Tasks that failed to meet an SLA. */
    splitByWaitTime?: string;
}
export interface WorkspaceStatisticsContext {
    /**
     * Fetch a WorkspaceStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkspaceStatisticsInstance) => any): Promise<WorkspaceStatisticsInstance>;
    /**
     * Fetch a WorkspaceStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceStatisticsInstance
     */
    fetch(params: WorkspaceStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkspaceStatisticsInstance) => any): Promise<WorkspaceStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WorkspaceStatisticsContextSolution {
    workspaceSid: string;
}
export declare class WorkspaceStatisticsContextImpl implements WorkspaceStatisticsContext {
    protected _version: V1;
    protected _solution: WorkspaceStatisticsContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string);
    fetch(params?: WorkspaceStatisticsContextFetchOptions | ((error: Error | null, item?: WorkspaceStatisticsInstance) => any), callback?: (error: Error | null, item?: WorkspaceStatisticsInstance) => any): Promise<WorkspaceStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WorkspaceStatisticsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WorkspaceStatisticsResource {
    realtime: any;
    cumulative: any;
    account_sid: string;
    workspace_sid: string;
    url: string;
}
export declare class WorkspaceStatisticsInstance {
    protected _version: V1;
    protected _solution: WorkspaceStatisticsContextSolution;
    protected _context?: WorkspaceStatisticsContext;
    constructor(_version: V1, payload: WorkspaceStatisticsResource, workspaceSid: string);
    /**
     * An object that contains the real-time statistics for the Workspace.
     */
    realtime: any;
    /**
     * An object that contains the cumulative statistics for the Workspace.
     */
    cumulative: any;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Workspace resource.
     */
    accountSid: string;
    /**
     * The SID of the Workspace.
     */
    workspaceSid: string;
    /**
     * The absolute URL of the Workspace statistics resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a WorkspaceStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkspaceStatisticsInstance) => any): Promise<WorkspaceStatisticsInstance>;
    /**
     * Fetch a WorkspaceStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceStatisticsInstance
     */
    fetch(params: WorkspaceStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkspaceStatisticsInstance) => any): Promise<WorkspaceStatisticsInstance>;
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
export interface WorkspaceStatisticsSolution {
    workspaceSid: string;
}
export interface WorkspaceStatisticsListInstance {
    _version: V1;
    _solution: WorkspaceStatisticsSolution;
    _uri: string;
    (): WorkspaceStatisticsContext;
    get(): WorkspaceStatisticsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WorkspaceStatisticsListInstance(version: V1, workspaceSid: string): WorkspaceStatisticsListInstance;
export {};
