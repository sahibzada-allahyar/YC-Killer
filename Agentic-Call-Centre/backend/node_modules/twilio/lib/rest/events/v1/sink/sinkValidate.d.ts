/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
/**
 * Options to pass to create a SinkValidateInstance
 */
export interface SinkValidateListInstanceCreateOptions {
    /** A 34 character string that uniquely identifies the test event for a Sink being validated. */
    testId: string;
}
export interface SinkValidateSolution {
    sid: string;
}
export interface SinkValidateListInstance {
    _version: V1;
    _solution: SinkValidateSolution;
    _uri: string;
    /**
     * Create a SinkValidateInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SinkValidateInstance
     */
    create(params: SinkValidateListInstanceCreateOptions, callback?: (error: Error | null, item?: SinkValidateInstance) => any): Promise<SinkValidateInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SinkValidateListInstance(version: V1, sid: string): SinkValidateListInstance;
interface SinkValidateResource {
    result: string;
}
export declare class SinkValidateInstance {
    protected _version: V1;
    constructor(_version: V1, payload: SinkValidateResource, sid: string);
    /**
     * Feedback indicating whether the given Sink was validated.
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
