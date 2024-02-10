/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
export type RecordingRecordingMode = "do-not-record" | "record-from-ringing" | "record-from-answer" | "record-from-ringing-dual" | "record-from-answer-dual";
export type RecordingRecordingTrim = "trim-silence" | "do-not-trim";
/**
 * Options to pass to update a RecordingInstance
 */
export interface RecordingContextUpdateOptions {
    /**  */
    mode?: RecordingRecordingMode;
    /**  */
    trim?: RecordingRecordingTrim;
}
export interface RecordingContext {
    /**
     * Fetch a RecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingInstance
     */
    fetch(callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Update a RecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingInstance
     */
    update(callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Update a RecordingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingInstance
     */
    update(params: RecordingContextUpdateOptions, callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface RecordingContextSolution {
    trunkSid: string;
}
export declare class RecordingContextImpl implements RecordingContext {
    protected _version: V1;
    protected _solution: RecordingContextSolution;
    protected _uri: string;
    constructor(_version: V1, trunkSid: string);
    fetch(callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    update(params?: RecordingContextUpdateOptions | ((error: Error | null, item?: RecordingInstance) => any), callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): RecordingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface RecordingResource {
    mode: RecordingRecordingMode;
    trim: RecordingRecordingTrim;
}
export declare class RecordingInstance {
    protected _version: V1;
    protected _solution: RecordingContextSolution;
    protected _context?: RecordingContext;
    constructor(_version: V1, payload: RecordingResource, trunkSid: string);
    mode: RecordingRecordingMode;
    trim: RecordingRecordingTrim;
    private get _proxy();
    /**
     * Fetch a RecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingInstance
     */
    fetch(callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Update a RecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingInstance
     */
    update(callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Update a RecordingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingInstance
     */
    update(params: RecordingContextUpdateOptions, callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        mode: RecordingRecordingMode;
        trim: RecordingRecordingTrim;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface RecordingSolution {
    trunkSid: string;
}
export interface RecordingListInstance {
    _version: V1;
    _solution: RecordingSolution;
    _uri: string;
    (): RecordingContext;
    get(): RecordingContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RecordingListInstance(version: V1, trunkSid: string): RecordingListInstance;
export {};
