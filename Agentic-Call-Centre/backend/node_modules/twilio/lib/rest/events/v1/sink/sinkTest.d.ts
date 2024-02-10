/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
export interface SinkTestSolution {
    sid: string;
}
export interface SinkTestListInstance {
    _version: V1;
    _solution: SinkTestSolution;
    _uri: string;
    /**
     * Create a SinkTestInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SinkTestInstance
     */
    create(callback?: (error: Error | null, item?: SinkTestInstance) => any): Promise<SinkTestInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SinkTestListInstance(version: V1, sid: string): SinkTestListInstance;
interface SinkTestResource {
    result: string;
}
export declare class SinkTestInstance {
    protected _version: V1;
    constructor(_version: V1, payload: SinkTestResource, sid: string);
    /**
     * Feedback indicating whether the test event was generated.
     */
    result: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        result: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
