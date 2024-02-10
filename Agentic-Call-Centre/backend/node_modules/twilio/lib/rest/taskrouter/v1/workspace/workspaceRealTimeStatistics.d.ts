/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
/**
 * Options to pass to fetch a WorkspaceRealTimeStatisticsInstance
 */
export interface WorkspaceRealTimeStatisticsContextFetchOptions {
    /** Only calculate real-time statistics on this TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
}
export interface WorkspaceRealTimeStatisticsContext {
    /**
     * Fetch a WorkspaceRealTimeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceRealTimeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkspaceRealTimeStatisticsInstance) => any): Promise<WorkspaceRealTimeStatisticsInstance>;
    /**
     * Fetch a WorkspaceRealTimeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceRealTimeStatisticsInstance
     */
    fetch(params: WorkspaceRealTimeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkspaceRealTimeStatisticsInstance) => any): Promise<WorkspaceRealTimeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WorkspaceRealTimeStatisticsContextSolution {
    workspaceSid: string;
}
export declare class WorkspaceRealTimeStatisticsContextImpl implements WorkspaceRealTimeStatisticsContext {
    protected _version: V1;
    protected _solution: WorkspaceRealTimeStatisticsContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string);
    fetch(params?: WorkspaceRealTimeStatisticsContextFetchOptions | ((error: Error | null, item?: WorkspaceRealTimeStatisticsInstance) => any), callback?: (error: Error | null, item?: WorkspaceRealTimeStatisticsInstance) => any): Promise<WorkspaceRealTimeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WorkspaceRealTimeStatisticsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WorkspaceRealTimeStatisticsResource {
    account_sid: string;
    activity_statistics: Array<any>;
    longest_task_waiting_age: number;
    longest_task_waiting_sid: string;
    tasks_by_priority: any;
    tasks_by_status: any;
    total_tasks: number;
    total_workers: number;
    workspace_sid: string;
    url: string;
}
export declare class WorkspaceRealTimeStatisticsInstance {
    protected _version: V1;
    protected _solution: WorkspaceRealTimeStatisticsContextSolution;
    protected _context?: WorkspaceRealTimeStatisticsContext;
    constructor(_version: V1, payload: WorkspaceRealTimeStatisticsResource, workspaceSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Workspace resource.
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
     * The number of Tasks by priority. For example: `{\"0\": \"10\", \"99\": \"5\"}` shows 10 Tasks at priority 0 and 5 at priority 99.
     */
    tasksByPriority: any;
    /**
     * The number of Tasks by their current status. For example: `{\"pending\": \"1\", \"reserved\": \"3\", \"assigned\": \"2\", \"completed\": \"5\"}`.
     */
    tasksByStatus: any;
    /**
     * The total number of Tasks.
     */
    totalTasks: number;
    /**
     * The total number of Workers in the Workspace.
     */
    totalWorkers: number;
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
     * Fetch a WorkspaceRealTimeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceRealTimeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkspaceRealTimeStatisticsInstance) => any): Promise<WorkspaceRealTimeStatisticsInstance>;
    /**
     * Fetch a WorkspaceRealTimeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkspaceRealTimeStatisticsInstance
     */
    fetch(params: WorkspaceRealTimeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkspaceRealTimeStatisticsInstance) => any): Promise<WorkspaceRealTimeStatisticsInstance>;
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
        tasksByPriority: any;
        tasksByStatus: any;
        totalTasks: number;
        totalWorkers: number;
        workspaceSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WorkspaceRealTimeStatisticsSolution {
    workspaceSid: string;
}
export interface WorkspaceRealTimeStatisticsListInstance {
    _version: V1;
    _solution: WorkspaceRealTimeStatisticsSolution;
    _uri: string;
    (): WorkspaceRealTimeStatisticsContext;
    get(): WorkspaceRealTimeStatisticsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WorkspaceRealTimeStatisticsListInstance(version: V1, workspaceSid: string): WorkspaceRealTimeStatisticsListInstance;
export {};
