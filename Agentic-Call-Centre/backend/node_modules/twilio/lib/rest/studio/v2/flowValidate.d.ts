/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../V2";
export type FlowValidateStatus = "draft" | "published";
/**
 * Options to pass to update a FlowValidateInstance
 */
export interface FlowValidateListInstanceUpdateOptions {
    /** The string that you assigned to describe the Flow. */
    friendlyName: string;
    /**  */
    status: FlowValidateStatus;
    /** JSON representation of flow definition. */
    definition: any;
    /** Description of change made in the revision. */
    commitMessage?: string;
}
export interface FlowValidateSolution {
}
export interface FlowValidateListInstance {
    _version: V2;
    _solution: FlowValidateSolution;
    _uri: string;
    /**
     * Update a FlowValidateInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlowValidateInstance
     */
    update(params: FlowValidateListInstanceUpdateOptions, callback?: (error: Error | null, item?: FlowValidateInstance) => any): Promise<FlowValidateInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function FlowValidateListInstance(version: V2): FlowValidateListInstance;
interface FlowValidateResource {
    valid: boolean;
}
export declare class FlowValidateInstance {
    protected _version: V2;
    constructor(_version: V2, payload: FlowValidateResource);
    /**
     * Boolean if the flow definition is valid.
     */
    valid: boolean;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        valid: boolean;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
