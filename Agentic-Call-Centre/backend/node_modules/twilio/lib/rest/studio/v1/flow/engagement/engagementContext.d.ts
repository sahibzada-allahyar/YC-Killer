/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
export interface EngagementContextContext {
    /**
     * Fetch a EngagementContextInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EngagementContextInstance
     */
    fetch(callback?: (error: Error | null, item?: EngagementContextInstance) => any): Promise<EngagementContextInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface EngagementContextContextSolution {
    flowSid: string;
    engagementSid: string;
}
export declare class EngagementContextContextImpl implements EngagementContextContext {
    protected _version: V1;
    protected _solution: EngagementContextContextSolution;
    protected _uri: string;
    constructor(_version: V1, flowSid: string, engagementSid: string);
    fetch(callback?: (error: Error | null, item?: EngagementContextInstance) => any): Promise<EngagementContextInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): EngagementContextContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface EngagementContextResource {
    account_sid: string;
    context: any;
    engagement_sid: string;
    flow_sid: string;
    url: string;
}
export declare class EngagementContextInstance {
    protected _version: V1;
    protected _solution: EngagementContextContextSolution;
    protected _context?: EngagementContextContext;
    constructor(_version: V1, payload: EngagementContextResource, flowSid: string, engagementSid: string);
    /**
     * The SID of the Account.
     */
    accountSid: string;
    /**
     * As your flow executes, we save the state in what\'s called the Flow Context. Any data in the flow context can be accessed by your widgets as variables, either in configuration fields or in text areas as variable substitution.
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
     * The URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a EngagementContextInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EngagementContextInstance
     */
    fetch(callback?: (error: Error | null, item?: EngagementContextInstance) => any): Promise<EngagementContextInstance>;
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
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface EngagementContextSolution {
    flowSid: string;
    engagementSid: string;
}
export interface EngagementContextListInstance {
    _version: V1;
    _solution: EngagementContextSolution;
    _uri: string;
    (): EngagementContextContext;
    get(): EngagementContextContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function EngagementContextListInstance(version: V1, flowSid: string, engagementSid: string): EngagementContextListInstance;
export {};
