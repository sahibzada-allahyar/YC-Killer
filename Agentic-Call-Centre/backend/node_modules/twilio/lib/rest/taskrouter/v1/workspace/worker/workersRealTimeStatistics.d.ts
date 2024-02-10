/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
/**
 * Options to pass to fetch a WorkersRealTimeStatisticsInstance
 */
export interface WorkersRealTimeStatisticsContextFetchOptions {
    /** Only calculate real-time statistics on this TaskChannel. Can be the TaskChannel\'s SID or its `unique_name`, such as `voice`, `sms`, or `default`. */
    taskChannel?: string;
}
export interface WorkersRealTimeStatisticsContext {
    /**
     * Fetch a WorkersRealTimeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkersRealTimeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkersRealTimeStatisticsInstance) => any): Promise<WorkersRealTimeStatisticsInstance>;
    /**
     * Fetch a WorkersRealTimeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkersRealTimeStatisticsInstance
     */
    fetch(params: WorkersRealTimeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkersRealTimeStatisticsInstance) => any): Promise<WorkersRealTimeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WorkersRealTimeStatisticsContextSolution {
    workspaceSid: string;
}
export declare class WorkersRealTimeStatisticsContextImpl implements WorkersRealTimeStatisticsContext {
    protected _version: V1;
    protected _solution: WorkersRealTimeStatisticsContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string);
    fetch(params?: WorkersRealTimeStatisticsContextFetchOptions | ((error: Error | null, item?: WorkersRealTimeStatisticsInstance) => any), callback?: (error: Error | null, item?: WorkersRealTimeStatisticsInstance) => any): Promise<WorkersRealTimeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WorkersRealTimeStatisticsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WorkersRealTimeStatisticsResource {
    account_sid: string;
    activity_statistics: Array<any>;
    total_workers: number;
    workspace_sid: string;
    url: string;
}
export declare class WorkersRealTimeStatisticsInstance {
    protected _version: V1;
    protected _solution: WorkersRealTimeStatisticsContextSolution;
    protected _context?: WorkersRealTimeStatisticsContext;
    constructor(_version: V1, payload: WorkersRealTimeStatisticsResource, workspaceSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Worker resource.
     */
    accountSid: string;
    /**
     * The number of current Workers by Activity.
     */
    activityStatistics: Array<any>;
    /**
     * The total number of Workers.
     */
    totalWorkers: number;
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
     * Fetch a WorkersRealTimeStatisticsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkersRealTimeStatisticsInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkersRealTimeStatisticsInstance) => any): Promise<WorkersRealTimeStatisticsInstance>;
    /**
     * Fetch a WorkersRealTimeStatisticsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkersRealTimeStatisticsInstance
     */
    fetch(params: WorkersRealTimeStatisticsContextFetchOptions, callback?: (error: Error | null, item?: WorkersRealTimeStatisticsInstance) => any): Promise<WorkersRealTimeStatisticsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        activityStatistics: any[];
        totalWorkers: number;
        workspaceSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WorkersRealTimeStatisticsSolution {
    workspaceSid: string;
}
export interface WorkersRealTimeStatisticsListInstance {
    _version: V1;
    _solution: WorkersRealTimeStatisticsSolution;
    _uri: string;
    (): WorkersRealTimeStatisticsContext;
    get(): WorkersRealTimeStatisticsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WorkersRealTimeStatisticsListInstance(version: V1, workspaceSid: string): WorkersRealTimeStatisticsListInstance;
export {};
