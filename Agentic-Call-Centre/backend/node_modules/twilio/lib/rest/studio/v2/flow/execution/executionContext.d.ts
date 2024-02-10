/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../../../V2";
export interface ExecutionContextContext {
    /**
     * Fetch a ExecutionContextInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExecutionContextInstance
     */
    fetch(callback?: (error: Error | null, item?: ExecutionContextInstance) => any): Promise<ExecutionContextInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ExecutionContextContextSolution {
    flowSid: string;
    executionSid: string;
}
export declare class ExecutionContextContextImpl implements ExecutionContextContext {
    protected _version: V2;
    protected _solution: ExecutionContextContextSolution;
    protected _uri: string;
    constructor(_version: V2, flowSid: string, executionSid: string);
    fetch(callback?: (error: Error | null, item?: ExecutionContextInstance) => any): Promise<ExecutionContextInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ExecutionContextContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ExecutionContextResource {
    account_sid: string;
    context: any;
    flow_sid: string;
    execution_sid: string;
    url: string;
}
export declare class ExecutionContextInstance {
    protected _version: V2;
    protected _solution: ExecutionContextContextSolution;
    protected _context?: ExecutionContextContext;
    constructor(_version: V2, payload: ExecutionContextResource, flowSid: string, executionSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the ExecutionContext resource.
     */
    accountSid: string;
    /**
     * The current state of the Flow\'s Execution. As a flow executes, we save its state in this context. We save data that your widgets can access as variables in configuration fields or in text areas as variable substitution.
     */
    context: any;
    /**
     * The SID of the Flow.
     */
    flowSid: string;
    /**
     * The SID of the context\'s Execution resource.
     */
    executionSid: string;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a ExecutionContextInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExecutionContextInstance
     */
    fetch(callback?: (error: Error | null, item?: ExecutionContextInstance) => any): Promise<ExecutionContextInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        context: any;
        flowSid: string;
        executionSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ExecutionContextSolution {
    flowSid: string;
    executionSid: string;
}
export interface ExecutionContextListInstance {
    _version: V2;
    _solution: ExecutionContextSolution;
    _uri: string;
    (): ExecutionContextContext;
    get(): ExecutionContextContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ExecutionContextListInstance(version: V2, flowSid: string, executionSid: string): ExecutionContextListInstance;
export {};
