/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../../../../V2";
export interface ExecutionStepContextContext {
    /**
     * Fetch a ExecutionStepContextInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExecutionStepContextInstance
     */
    fetch(callback?: (error: Error | null, item?: ExecutionStepContextInstance) => any): Promise<ExecutionStepContextInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ExecutionStepContextContextSolution {
    flowSid: string;
    executionSid: string;
    stepSid: string;
}
export declare class ExecutionStepContextContextImpl implements ExecutionStepContextContext {
    protected _version: V2;
    protected _solution: ExecutionStepContextContextSolution;
    protected _uri: string;
    constructor(_version: V2, flowSid: string, executionSid: string, stepSid: string);
    fetch(callback?: (error: Error | null, item?: ExecutionStepContextInstance) => any): Promise<ExecutionStepContextInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ExecutionStepContextContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ExecutionStepContextResource {
    account_sid: string;
    context: any;
    execution_sid: string;
    flow_sid: string;
    step_sid: string;
    url: string;
}
export declare class ExecutionStepContextInstance {
    protected _version: V2;
    protected _solution: ExecutionStepContextContextSolution;
    protected _context?: ExecutionStepContextContext;
    constructor(_version: V2, payload: ExecutionStepContextResource, flowSid: string, executionSid: string, stepSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the ExecutionStepContext resource.
     */
    accountSid: string;
    /**
     * The current state of the Flow\'s Execution. As a flow executes, we save its state in this context. We save data that your widgets can access as variables in configuration fields or in text areas as variable substitution.
     */
    context: any;
    /**
     * The SID of the context\'s Execution resource.
     */
    executionSid: string;
    /**
     * The SID of the Flow.
     */
    flowSid: string;
    /**
     * The SID of the Step that the context is associated with.
     */
    stepSid: string;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a ExecutionStepContextInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ExecutionStepContextInstance
     */
    fetch(callback?: (error: Error | null, item?: ExecutionStepContextInstance) => any): Promise<ExecutionStepContextInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        context: any;
        executionSid: string;
        flowSid: string;
        stepSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ExecutionStepContextSolution {
    flowSid: string;
    executionSid: string;
    stepSid: string;
}
export interface ExecutionStepContextListInstance {
    _version: V2;
    _solution: ExecutionStepContextSolution;
    _uri: string;
    (): ExecutionStepContextContext;
    get(): ExecutionStepContextContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ExecutionStepContextListInstance(version: V2, flowSid: string, executionSid: string, stepSid: string): ExecutionStepContextListInstance;
export {};
