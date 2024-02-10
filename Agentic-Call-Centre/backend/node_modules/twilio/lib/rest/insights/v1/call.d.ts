/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
import { AnnotationListInstance } from "./call/annotation";
import { CallSummaryListInstance } from "./call/callSummary";
import { EventListInstance } from "./call/event";
import { MetricListInstance } from "./call/metric";
export interface CallContext {
    annotation: AnnotationListInstance;
    summary: CallSummaryListInstance;
    events: EventListInstance;
    metrics: MetricListInstance;
    /**
     * Fetch a CallInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CallInstance
     */
    fetch(callback?: (error: Error | null, item?: CallInstance) => any): Promise<CallInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface CallContextSolution {
    sid: string;
}
export declare class CallContextImpl implements CallContext {
    protected _version: V1;
    protected _solution: CallContextSolution;
    protected _uri: string;
    protected _annotation?: AnnotationListInstance;
    protected _summary?: CallSummaryListInstance;
    protected _events?: EventListInstance;
    protected _metrics?: MetricListInstance;
    constructor(_version: V1, sid: string);
    get annotation(): AnnotationListInstance;
    get summary(): CallSummaryListInstance;
    get events(): EventListInstance;
    get metrics(): MetricListInstance;
    fetch(callback?: (error: Error | null, item?: CallInstance) => any): Promise<CallInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): CallContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface CallResource {
    sid: string;
    url: string;
    links: Record<string, string>;
}
export declare class CallInstance {
    protected _version: V1;
    protected _solution: CallContextSolution;
    protected _context?: CallContext;
    constructor(_version: V1, payload: CallResource, sid?: string);
    sid: string;
    url: string;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a CallInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CallInstance
     */
    fetch(callback?: (error: Error | null, item?: CallInstance) => any): Promise<CallInstance>;
    /**
     * Access the annotation.
     */
    annotation(): AnnotationListInstance;
    /**
     * Access the summary.
     */
    summary(): CallSummaryListInstance;
    /**
     * Access the events.
     */
    events(): EventListInstance;
    /**
     * Access the metrics.
     */
    metrics(): MetricListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CallSolution {
}
export interface CallListInstance {
    _version: V1;
    _solution: CallSolution;
    _uri: string;
    (sid: string): CallContext;
    get(sid: string): CallContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CallListInstance(version: V1): CallListInstance;
export {};
