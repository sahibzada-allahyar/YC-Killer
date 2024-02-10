/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export interface UsecaseSolution {
}
export interface UsecaseListInstance {
    _version: V1;
    _solution: UsecaseSolution;
    _uri: string;
    /**
     * Fetch a UsecaseInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UsecaseInstance
     */
    fetch(callback?: (error: Error | null, item?: UsecaseInstance) => any): Promise<UsecaseInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function UsecaseListInstance(version: V1): UsecaseListInstance;
interface UsecaseResource {
    usecases: Array<any>;
}
export declare class UsecaseInstance {
    protected _version: V1;
    constructor(_version: V1, payload: UsecaseResource);
    /**
     * Human readable use case details (usecase, description and purpose) of Messaging Service Use Cases.
     */
    usecases: Array<any>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        usecases: any[];
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
