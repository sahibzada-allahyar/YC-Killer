/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../V2";
/**
 * Options to pass to create a SafelistInstance
 */
export interface SafelistListInstanceCreateOptions {
    /** The phone number to be added in SafeList. Phone numbers must be in [E.164 format](https://www.twilio.com/docs/glossary/what-e164). */
    phoneNumber: string;
}
export interface SafelistContext {
    /**
     * Remove a SafelistInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SafelistInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SafelistInstance
     */
    fetch(callback?: (error: Error | null, item?: SafelistInstance) => any): Promise<SafelistInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SafelistContextSolution {
    phoneNumber: string;
}
export declare class SafelistContextImpl implements SafelistContext {
    protected _version: V2;
    protected _solution: SafelistContextSolution;
    protected _uri: string;
    constructor(_version: V2, phoneNumber: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SafelistInstance) => any): Promise<SafelistInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SafelistContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SafelistResource {
    sid: string;
    phone_number: string;
    url: string;
}
export declare class SafelistInstance {
    protected _version: V2;
    protected _solution: SafelistContextSolution;
    protected _context?: SafelistContext;
    constructor(_version: V2, payload: SafelistResource, phoneNumber?: string);
    /**
     * The unique string that we created to identify the SafeList resource.
     */
    sid: string;
    /**
     * The phone number in SafeList.
     */
    phoneNumber: string;
    /**
     * The absolute URL of the SafeList resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a SafelistInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SafelistInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SafelistInstance
     */
    fetch(callback?: (error: Error | null, item?: SafelistInstance) => any): Promise<SafelistInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        phoneNumber: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SafelistSolution {
}
export interface SafelistListInstance {
    _version: V2;
    _solution: SafelistSolution;
    _uri: string;
    (phoneNumber: string): SafelistContext;
    get(phoneNumber: string): SafelistContext;
    /**
     * Create a SafelistInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SafelistInstance
     */
    create(params: SafelistListInstanceCreateOptions, callback?: (error: Error | null, item?: SafelistInstance) => any): Promise<SafelistInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SafelistListInstance(version: V2): SafelistListInstance;
export {};
