/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../../V1";
export interface StepContextContext {
    /**
     * Fetch a StepContextInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed StepContextInstance
     */
    fetch(callback?: (error: Error | null, item?: StepContextInstance) => any): Promise<StepContextInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface StepContextContextSolution {
    flowSid: string;
    engagementSid: string;
    stepSid: string;
}
export declare class StepContextContextImpl implements StepContextContext {
    protected _version: V1;
    protected _solution: StepContextContextSolution;
    protected _uri: string;
    constructor(_version: V1, flowSid: string, engagementSid: string, stepSid: string);
    fetch(callback?: (error: Error | null, item?: StepContextInstance) => any): Promise<StepContextInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): StepContextContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface StepContextResource {
    account_sid: string;
    context: any;
    engagement_sid: string;
    flow_sid: string;
    step_sid: string;
    url: string;
}
export declare class StepContextInstance {
    protected _version: V1;
    protected _solution: StepContextContextSolution;
    protected _context?: StepContextContext;
    constructor(_version: V1, payload: StepContextResource, flowSid: string, engagementSid: string, stepSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the StepContext resource.
     */
    accountSid: string;
    /**
     * The current state of the Flow\'s Execution. As a flow executes, we save its state in this context. We save data that your widgets can access as variables in configuration fields or in text areas as variable substitution.
     */
    context: any;
    /**
     * The SID of the Engagement.
     */
    engagementSid: string;
    /**
     * The SID of the Flow.
     */
    flowSid: string;
    /**
     * The SID of the Step the context is associated with.
     */
    stepSid: string;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a StepContextInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed StepContextInstance
     */
    fetch(callback?: (error: Error | null, item?: StepContextInstance) => any): Promise<StepContextInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        context: any;
        engagementSid: string;
        flowSid: string;
        stepSid: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface StepContextSolution {
    flowSid: string;
    engagementSid: string;
    stepSid: string;
}
export interface StepContextListInstance {
    _version: V1;
    _solution: StepContextSolution;
    _uri: string;
    (): StepContextContext;
    get(): StepContextContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function StepContextListInstance(version: V1, flowSid: string, engagementSid: string, stepSid: string): StepContextListInstance;
export {};
