/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export interface ArchivedCallContext {
    /**
     * Remove a ArchivedCallInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ArchivedCallContextSolution {
    date: Date;
    sid: string;
}
export declare class ArchivedCallContextImpl implements ArchivedCallContext {
    protected _version: V1;
    protected _solution: ArchivedCallContextSolution;
    protected _uri: string;
    constructor(_version: V1, date: Date, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ArchivedCallContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ArchivedCallSolution {
}
export interface ArchivedCallListInstance {
    _version: V1;
    _solution: ArchivedCallSolution;
    _uri: string;
    (date: Date, sid: string): ArchivedCallContext;
    get(date: Date, sid: string): ArchivedCallContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ArchivedCallListInstance(version: V1): ArchivedCallListInstance;
