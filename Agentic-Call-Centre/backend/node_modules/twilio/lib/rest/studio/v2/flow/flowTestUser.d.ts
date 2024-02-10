/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../../V2";
/**
 * Options to pass to update a FlowTestUserInstance
 */
export interface FlowTestUserContextUpdateOptions {
    /** List of test user identities that can test draft versions of the flow. */
    testUsers: Array<string>;
}
export interface FlowTestUserContext {
    /**
     * Fetch a FlowTestUserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlowTestUserInstance
     */
    fetch(callback?: (error: Error | null, item?: FlowTestUserInstance) => any): Promise<FlowTestUserInstance>;
    /**
     * Update a FlowTestUserInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlowTestUserInstance
     */
    update(params: FlowTestUserContextUpdateOptions, callback?: (error: Error | null, item?: FlowTestUserInstance) => any): Promise<FlowTestUserInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface FlowTestUserContextSolution {
    sid: string;
}
export declare class FlowTestUserContextImpl implements FlowTestUserContext {
    protected _version: V2;
    protected _solution: FlowTestUserContextSolution;
    protected _uri: string;
    constructor(_version: V2, sid: string);
    fetch(callback?: (error: Error | null, item?: FlowTestUserInstance) => any): Promise<FlowTestUserInstance>;
    update(params: FlowTestUserContextUpdateOptions, callback?: (error: Error | null, item?: FlowTestUserInstance) => any): Promise<FlowTestUserInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): FlowTestUserContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface FlowTestUserResource {
    sid: string;
    test_users: Array<string>;
    url: string;
}
export declare class FlowTestUserInstance {
    protected _version: V2;
    protected _solution: FlowTestUserContextSolution;
    protected _context?: FlowTestUserContext;
    constructor(_version: V2, payload: FlowTestUserResource, sid: string);
    /**
     * Unique identifier of the flow.
     */
    sid: string;
    /**
     * List of test user identities that can test draft versions of the flow.
     */
    testUsers: Array<string>;
    /**
     * The URL of this resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a FlowTestUserInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlowTestUserInstance
     */
    fetch(callback?: (error: Error | null, item?: FlowTestUserInstance) => any): Promise<FlowTestUserInstance>;
    /**
     * Update a FlowTestUserInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlowTestUserInstance
     */
    update(params: FlowTestUserContextUpdateOptions, callback?: (error: Error | null, item?: FlowTestUserInstance) => any): Promise<FlowTestUserInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        testUsers: string[];
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface FlowTestUserSolution {
    sid: string;
}
export interface FlowTestUserListInstance {
    _version: V2;
    _solution: FlowTestUserSolution;
    _uri: string;
    (): FlowTestUserContext;
    get(): FlowTestUserContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function FlowTestUserListInstance(version: V2, sid: string): FlowTestUserListInstance;
export {};
