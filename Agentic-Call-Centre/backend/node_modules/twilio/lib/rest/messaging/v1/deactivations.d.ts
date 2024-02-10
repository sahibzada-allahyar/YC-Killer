/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to fetch a DeactivationsInstance
 */
export interface DeactivationsContextFetchOptions {
    /** The request will return a list of all United States Phone Numbers that were deactivated on the day specified by this parameter. This date should be specified in YYYY-MM-DD format. */
    date?: Date;
}
export interface DeactivationsContext {
    /**
     * Fetch a DeactivationsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeactivationsInstance
     */
    fetch(callback?: (error: Error | null, item?: DeactivationsInstance) => any): Promise<DeactivationsInstance>;
    /**
     * Fetch a DeactivationsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeactivationsInstance
     */
    fetch(params: DeactivationsContextFetchOptions, callback?: (error: Error | null, item?: DeactivationsInstance) => any): Promise<DeactivationsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface DeactivationsContextSolution {
}
export declare class DeactivationsContextImpl implements DeactivationsContext {
    protected _version: V1;
    protected _solution: DeactivationsContextSolution;
    protected _uri: string;
    constructor(_version: V1);
    fetch(params?: DeactivationsContextFetchOptions | ((error: Error | null, item?: DeactivationsInstance) => any), callback?: (error: Error | null, item?: DeactivationsInstance) => any): Promise<DeactivationsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): DeactivationsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface DeactivationsResource {
    redirect_to: string;
}
export declare class DeactivationsInstance {
    protected _version: V1;
    protected _solution: DeactivationsContextSolution;
    protected _context?: DeactivationsContext;
    constructor(_version: V1, payload: DeactivationsResource);
    /**
     * Returns an authenticated url that redirects to a file containing the deactivated numbers for the requested day. This url is valid for up to two minutes.
     */
    redirectTo: string;
    private get _proxy();
    /**
     * Fetch a DeactivationsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeactivationsInstance
     */
    fetch(callback?: (error: Error | null, item?: DeactivationsInstance) => any): Promise<DeactivationsInstance>;
    /**
     * Fetch a DeactivationsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeactivationsInstance
     */
    fetch(params: DeactivationsContextFetchOptions, callback?: (error: Error | null, item?: DeactivationsInstance) => any): Promise<DeactivationsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        redirectTo: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DeactivationsSolution {
}
export interface DeactivationsListInstance {
    _version: V1;
    _solution: DeactivationsSolution;
    _uri: string;
    (): DeactivationsContext;
    get(): DeactivationsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DeactivationsListInstance(version: V1): DeactivationsListInstance;
export {};
