/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to fetch a InsightsUserRolesInstance
 */
export interface InsightsUserRolesContextFetchOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
}
export interface InsightsUserRolesContext {
    /**
     * Fetch a InsightsUserRolesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsUserRolesInstance
     */
    fetch(callback?: (error: Error | null, item?: InsightsUserRolesInstance) => any): Promise<InsightsUserRolesInstance>;
    /**
     * Fetch a InsightsUserRolesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsUserRolesInstance
     */
    fetch(params: InsightsUserRolesContextFetchOptions, callback?: (error: Error | null, item?: InsightsUserRolesInstance) => any): Promise<InsightsUserRolesInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface InsightsUserRolesContextSolution {
}
export declare class InsightsUserRolesContextImpl implements InsightsUserRolesContext {
    protected _version: V1;
    protected _solution: InsightsUserRolesContextSolution;
    protected _uri: string;
    constructor(_version: V1);
    fetch(params?: InsightsUserRolesContextFetchOptions | ((error: Error | null, item?: InsightsUserRolesInstance) => any), callback?: (error: Error | null, item?: InsightsUserRolesInstance) => any): Promise<InsightsUserRolesInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): InsightsUserRolesContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface InsightsUserRolesResource {
    roles: Array<string>;
    url: string;
}
export declare class InsightsUserRolesInstance {
    protected _version: V1;
    protected _solution: InsightsUserRolesContextSolution;
    protected _context?: InsightsUserRolesContext;
    constructor(_version: V1, payload: InsightsUserRolesResource);
    /**
     * Flex Insights roles for the user
     */
    roles: Array<string>;
    url: string;
    private get _proxy();
    /**
     * Fetch a InsightsUserRolesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsUserRolesInstance
     */
    fetch(callback?: (error: Error | null, item?: InsightsUserRolesInstance) => any): Promise<InsightsUserRolesInstance>;
    /**
     * Fetch a InsightsUserRolesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsUserRolesInstance
     */
    fetch(params: InsightsUserRolesContextFetchOptions, callback?: (error: Error | null, item?: InsightsUserRolesInstance) => any): Promise<InsightsUserRolesInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        roles: string[];
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface InsightsUserRolesSolution {
}
export interface InsightsUserRolesListInstance {
    _version: V1;
    _solution: InsightsUserRolesSolution;
    _uri: string;
    (): InsightsUserRolesContext;
    get(): InsightsUserRolesContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InsightsUserRolesListInstance(version: V1): InsightsUserRolesListInstance;
export {};
