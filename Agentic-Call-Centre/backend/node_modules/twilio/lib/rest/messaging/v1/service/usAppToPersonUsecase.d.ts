/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
/**
 * Options to pass to fetch a UsAppToPersonUsecaseInstance
 */
export interface UsAppToPersonUsecaseListInstanceFetchOptions {
    /** The unique string to identify the A2P brand. */
    brandRegistrationSid?: string;
}
export interface UsAppToPersonUsecaseSolution {
    messagingServiceSid: string;
}
export interface UsAppToPersonUsecaseListInstance {
    _version: V1;
    _solution: UsAppToPersonUsecaseSolution;
    _uri: string;
    /**
     * Fetch a UsAppToPersonUsecaseInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UsAppToPersonUsecaseInstance
     */
    fetch(callback?: (error: Error | null, item?: UsAppToPersonUsecaseInstance) => any): Promise<UsAppToPersonUsecaseInstance>;
    /**
     * Fetch a UsAppToPersonUsecaseInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UsAppToPersonUsecaseInstance
     */
    fetch(params: UsAppToPersonUsecaseListInstanceFetchOptions, callback?: (error: Error | null, item?: UsAppToPersonUsecaseInstance) => any): Promise<UsAppToPersonUsecaseInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function UsAppToPersonUsecaseListInstance(version: V1, messagingServiceSid: string): UsAppToPersonUsecaseListInstance;
interface UsAppToPersonUsecaseResource {
    us_app_to_person_usecases: Array<any>;
}
export declare class UsAppToPersonUsecaseInstance {
    protected _version: V1;
    constructor(_version: V1, payload: UsAppToPersonUsecaseResource, messagingServiceSid: string);
    /**
     * Human readable name, code, description and post_approval_required (indicates whether or not post approval is required for this Use Case) of A2P Campaign Use Cases.
     */
    usAppToPersonUsecases: Array<any>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        usAppToPersonUsecases: any[];
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
