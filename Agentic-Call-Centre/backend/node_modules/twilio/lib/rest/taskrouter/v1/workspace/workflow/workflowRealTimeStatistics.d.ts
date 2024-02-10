/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
/**
 * Options to pass to fetch a WorkflowRealTimeStatisticsInstance
 */
export interface WorkflowRealTimeStatisticsContextFetchOptions {
    /** Only calculate real-time statistics on this TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
}
export interface WorkflowRealTimeStatisticsContext {
    /**
     * Fetch a WorkflowRealTimeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowRealTimeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkflowRealTimeStatisticsInstance) => any): Promise<WorkflowRealTimeStatisticsInstance>;
    /**
     * Fetch a WorkflowRealTimeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowRealTimeStatisticsInstance
     */
    fetch(params: WorkflowRealTimeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkflowRealTimeStatisticsInstance) => any): Promise<WorkflowRealTimeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WorkflowRealTimeStatisticsContextSolution {
    workspaceSid: string;
    workflowSid: string;
}
export declare class WorkflowRealTimeStatisticsContextImpl implements WorkflowRealTimeStatisticsContext {
    protected _version: V1;
    protected _solution: WorkflowRealTimeStatisticsContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string, workflowSid: string);
    fetch(params?: WorkflowRealTimeStatisticsContextFetchOptions | ((error: Error | null, item?: WorkflowRealTimeStatisticsInstance) => any), callback?: (error: Error | null, item?: WorkflowRealTimeStatisticsInstance) => any): Promise<WorkflowRealTimeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WorkflowRealTimeStatisticsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WorkflowRealTimeStatisticsResource {
    account_sid: string;
    longest_task_waiting_age: number;
    longest_task_waiting_sid: string;
    tasks_by_priority: any;
    tasks_by_status: any;
    total_tasks: number;
    workflow_sid: string;
    workspace_sid: string;
    url: string;
}
export declare class WorkflowRealTimeStatisticsInstance {
    protected _version: V1;
    protected _solution: WorkflowRealTimeStatisticsContextSolution;
    protected _context?: WorkflowRealTimeStatisticsContext;
    constructor(_version: V1, payload: WorkflowRealTimeStatisticsResource, workspaceSid: string, workflowSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Workflow resource.
     */
    accountSid: string;
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
     * Returns the list of Tasks that are being controlled by the Workflow with the specified SID value.
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
     * Fetch a WorkflowRealTimeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowRealTimeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkflowRealTimeStatisticsInstance) => any): Promise<WorkflowRealTimeStatisticsInstance>;
    /**
     * Fetch a WorkflowRealTimeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowRealTimeStatisticsInstance
     */
    fetch(params: WorkflowRealTimeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkflowRealTimeStatisticsInstance) => any): Promise<WorkflowRealTimeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        longestTaskWaitingAge: number;
        longestTaskWaitingSid: string;
        tasksByPriority: any;
        tasksByStatus: any;
        totalTasks: number;
        workflowSid: string;
        workspaceSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WorkflowRealTimeStatisticsSolution {
    workspaceSid: string;
    workflowSid: string;
}
export interface WorkflowRealTimeStatisticsListInstance {
    _version: V1;
    _solution: WorkflowRealTimeStatisticsSolution;
    _uri: string;
    (): WorkflowRealTimeStatisticsContext;
    get(): WorkflowRealTimeStatisticsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WorkflowRealTimeStatisticsListInstance(version: V1, workspaceSid: string, workflowSid: string): WorkflowRealTimeStatisticsListInstance;
export {};
