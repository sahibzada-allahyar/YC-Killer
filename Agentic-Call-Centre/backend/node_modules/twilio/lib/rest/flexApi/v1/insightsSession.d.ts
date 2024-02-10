/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to create a InsightsSessionInstance
 */
export interface InsightsSessionContextCreateOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
}
export interface InsightsSessionContext {
    /**
     * Create a InsightsSessionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsSessionInstance
     */
    create(callback?: (error: Error | null, item?: InsightsSessionInstance) => any): Promise<InsightsSessionInstance>;
    /**
     * Create a InsightsSessionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsSessionInstance
     */
    create(params: InsightsSessionContextCreateOptions, callback?: (error: Error | null, item?: InsightsSessionInstance) => any): Promise<InsightsSessionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface InsightsSessionContextSolution {
}
export declare class InsightsSessionContextImpl implements InsightsSessionContext {
    protected _version: V1;
    protected _solution: InsightsSessionContextSolution;
    protected _uri: string;
    constructor(_version: V1);
    create(params?: InsightsSessionContextCreateOptions | ((error: Error | null, item?: InsightsSessionInstance) => any), callback?: (error: Error | null, item?: InsightsSessionInstance) => any): Promise<InsightsSessionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): InsightsSessionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface InsightsSessionResource {
    workspace_id: string;
    session_expiry: string;
    session_id: string;
    base_url: string;
    url: string;
}
export declare class InsightsSessionInstance {
    protected _version: V1;
    protected _solution: InsightsSessionContextSolution;
    protected _context?: InsightsSessionContext;
    constructor(_version: V1, payload: InsightsSessionResource);
    /**
     * Unique ID to identify the user\'s workspace
     */
    workspaceId: string;
    /**
     * The session expiry date and time, given in ISO 8601 format.
     */
    sessionExpiry: string;
    /**
     * The unique ID for the session
     */
    sessionId: string;
    /**
     * The base URL to fetch reports and dashboards
     */
    baseUrl: string;
    /**
     * The URL of this resource.
     */
    url: string;
    private get _proxy();
    /**
     * Create a InsightsSessionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsSessionInstance
     */
    create(callback?: (error: Error | null, item?: InsightsSessionInstance) => any): Promise<InsightsSessionInstance>;
    /**
     * Create a InsightsSessionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsSessionInstance
     */
    create(params: InsightsSessionContextCreateOptions, callback?: (error: Error | null, item?: InsightsSessionInstance) => any): Promise<InsightsSessionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        workspaceId: string;
        sessionExpiry: string;
        sessionId: string;
        baseUrl: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface InsightsSessionSolution {
}
export interface InsightsSessionListInstance {
    _version: V1;
    _solution: InsightsSessionSolution;
    _uri: string;
    (): InsightsSessionContext;
    get(): InsightsSessionContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InsightsSessionListInstance(version: V1): InsightsSessionListInstance;
export {};
