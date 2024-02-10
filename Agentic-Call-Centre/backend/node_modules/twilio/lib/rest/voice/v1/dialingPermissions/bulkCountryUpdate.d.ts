/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
/**
 * Options to pass to create a BulkCountryUpdateInstance
 */
export interface BulkCountryUpdateListInstanceCreateOptions {
    /** URL encoded JSON array of update objects. example : `[ { \\\"iso_code\\\": \\\"GB\\\", \\\"low_risk_numbers_enabled\\\": \\\"true\\\", \\\"high_risk_special_numbers_enabled\\\":\\\"true\\\", \\\"high_risk_tollfraud_numbers_enabled\\\": \\\"false\\\" } ]` */
    updateRequest: string;
}
export interface BulkCountryUpdateSolution {
}
export interface BulkCountryUpdateListInstance {
    _version: V1;
    _solution: BulkCountryUpdateSolution;
    _uri: string;
    /**
     * Create a BulkCountryUpdateInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BulkCountryUpdateInstance
     */
    create(params: BulkCountryUpdateListInstanceCreateOptions, callback?: (error: Error | null, item?: BulkCountryUpdateInstance) => any): Promise<BulkCountryUpdateInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BulkCountryUpdateListInstance(version: V1): BulkCountryUpdateListInstance;
interface BulkCountryUpdateResource {
    update_count: number;
    update_request: string;
}
export declare class BulkCountryUpdateInstance {
    protected _version: V1;
    constructor(_version: V1, payload: BulkCountryUpdateResource);
    /**
     * The number of countries updated
     */
    updateCount: number;
    /**
     * A bulk update request to change voice dialing country permissions stored as a URL-encoded, JSON array of update objects. For example : `[ { \"iso_code\": \"GB\", \"low_risk_numbers_enabled\": \"true\", \"high_risk_special_numbers_enabled\":\"true\", \"high_risk_tollfraud_numbers_enabled\": \"false\" } ]`
     */
    updateRequest: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        updateCount: number;
        updateRequest: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
