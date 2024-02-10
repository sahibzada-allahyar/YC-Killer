/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
export type AnnotationAnsweredBy = "unknown_answered_by" | "human" | "machine";
export type AnnotationConnectivityIssue = "unknown_connectivity_issue" | "no_connectivity_issue" | "invalid_number" | "caller_id" | "dropped_call" | "number_reachability";
/**
 * Options to pass to update a AnnotationInstance
 */
export interface AnnotationContextUpdateOptions {
    /**  */
    answeredBy?: AnnotationAnsweredBy;
    /**  */
    connectivityIssue?: AnnotationConnectivityIssue;
    /** Specify if the call had any subjective quality issues. Possible values, one or more of `no_quality_issue`, `low_volume`, `choppy_robotic`, `echo`, `dtmf`, `latency`, `owa`, `static_noise`. Use comma separated values to indicate multiple quality issues for the same call. */
    qualityIssues?: string;
    /** A boolean flag to indicate if the call was a spam call. Use this to provide feedback on whether calls placed from your account were marked as spam, or if inbound calls received by your account were unwanted spam. Use `true` if the call was a spam call. */
    spam?: boolean;
    /** Specify the call score. This is of type integer. Use a range of 1-5 to indicate the call experience score, with the following mapping as a reference for rating the call [5: Excellent, 4: Good, 3 : Fair, 2 : Poor, 1: Bad]. */
    callScore?: number;
    /** Specify any comments pertaining to the call. `comment` has a maximum character limit of 100. Twilio does not treat this field as PII, so no PII should be included in the `comment`. */
    comment?: string;
    /** Associate this call with an incident or support ticket. The `incident` parameter is of type string with a maximum character limit of 100. Twilio does not treat this field as PII, so no PII should be included in `incident`. */
    incident?: string;
}
export interface AnnotationContext {
    /**
     * Fetch a AnnotationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AnnotationInstance
     */
    fetch(callback?: (error: Error | null, item?: AnnotationInstance) => any): Promise<AnnotationInstance>;
    /**
     * Update a AnnotationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AnnotationInstance
     */
    update(callback?: (error: Error | null, item?: AnnotationInstance) => any): Promise<AnnotationInstance>;
    /**
     * Update a AnnotationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AnnotationInstance
     */
    update(params: AnnotationContextUpdateOptions, callback?: (error: Error | null, item?: AnnotationInstance) => any): Promise<AnnotationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AnnotationContextSolution {
    callSid: string;
}
export declare class AnnotationContextImpl implements AnnotationContext {
    protected _version: V1;
    protected _solution: AnnotationContextSolution;
    protected _uri: string;
    constructor(_version: V1, callSid: string);
    fetch(callback?: (error: Error | null, item?: AnnotationInstance) => any): Promise<AnnotationInstance>;
    update(params?: AnnotationContextUpdateOptions | ((error: Error | null, item?: AnnotationInstance) => any), callback?: (error: Error | null, item?: AnnotationInstance) => any): Promise<AnnotationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AnnotationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AnnotationResource {
    call_sid: string;
    account_sid: string;
    answered_by: AnnotationAnsweredBy;
    connectivity_issue: AnnotationConnectivityIssue;
    quality_issues: Array<string>;
    spam: boolean;
    call_score: number;
    comment: string;
    incident: string;
    url: string;
}
export declare class AnnotationInstance {
    protected _version: V1;
    protected _solution: AnnotationContextSolution;
    protected _context?: AnnotationContext;
    constructor(_version: V1, payload: AnnotationResource, callSid: string);
    /**
     * The unique SID identifier of the Call.
     */
    callSid: string;
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    answeredBy: AnnotationAnsweredBy;
    connectivityIssue: AnnotationConnectivityIssue;
    /**
     * Specifies if the call had any subjective quality issues. Possible values are one or more of `no_quality_issue`, `low_volume`, `choppy_robotic`, `echo`, `dtmf`, `latency`, `owa`, or `static_noise`.
     */
    qualityIssues: Array<string>;
    /**
     * Specifies if the call was a spam call. Use this to provide feedback on whether calls placed from your account were marked as spam, or if inbound calls received by your account were unwanted spam. Is of type Boolean: true, false. Use true if the call was a spam call.
     */
    spam: boolean;
    /**
     * Specifies the Call Score, if available. This is of type integer. Use a range of 1-5 to indicate the call experience score, with the following mapping as a reference for rating the call [5: Excellent, 4: Good, 3 : Fair, 2 : Poor, 1: Bad].
     */
    callScore: number;
    /**
     * Specifies any comments pertaining to the call. Twilio does not treat this field as PII, so no PII should be included in comments.
     */
    comment: string;
    /**
     * Incident or support ticket associated with this call. The `incident` property is of type string with a maximum character limit of 100. Twilio does not treat this field as PII, so no PII should be included in `incident`.
     */
    incident: string;
    url: string;
    private get _proxy();
    /**
     * Fetch a AnnotationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AnnotationInstance
     */
    fetch(callback?: (error: Error | null, item?: AnnotationInstance) => any): Promise<AnnotationInstance>;
    /**
     * Update a AnnotationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AnnotationInstance
     */
    update(callback?: (error: Error | null, item?: AnnotationInstance) => any): Promise<AnnotationInstance>;
    /**
     * Update a AnnotationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AnnotationInstance
     */
    update(params: AnnotationContextUpdateOptions, callback?: (error: Error | null, item?: AnnotationInstance) => any): Promise<AnnotationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        callSid: string;
        accountSid: string;
        answeredBy: AnnotationAnsweredBy;
        connectivityIssue: AnnotationConnectivityIssue;
        qualityIssues: string[];
        spam: boolean;
        callScore: number;
        comment: string;
        incident: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AnnotationSolution {
    callSid: string;
}
export interface AnnotationListInstance {
    _version: V1;
    _solution: AnnotationSolution;
    _uri: string;
    (): AnnotationContext;
    get(): AnnotationContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AnnotationListInstance(version: V1, callSid: string): AnnotationListInstance;
export {};
