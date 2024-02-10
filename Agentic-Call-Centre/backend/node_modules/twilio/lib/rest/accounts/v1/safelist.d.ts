/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to create a SafelistInstance
 */
export interface SafelistListInstanceCreateOptions {
    /** The phone number to be added in SafeList. Phone numbers must be in [E.164 format](https://www.twilio.com/docs/glossary/what-e164). */
    phoneNumber: string;
}
/**
 * Options to pass to remove a SafelistInstance
 */
export interface SafelistListInstanceRemoveOptions {
    /** The phone number to be removed from SafeList. Phone numbers must be in [E.164 format](https://www.twilio.com/docs/glossary/what-e164). */
    phoneNumber?: string;
}
/**
 * Options to pass to fetch a SafelistInstance
 */
export interface SafelistListInstanceFetchOptions {
    /** The phone number to be fetched from SafeList. Phone numbers must be in [E.164 format](https://www.twilio.com/docs/glossary/what-e164). */
    phoneNumber?: string;
}
export interface SafelistSolution {
}
export interface SafelistListInstance {
    _version: V1;
    _solution: SafelistSolution;
    _uri: string;
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
     * Remove a SafelistInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a SafelistInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SafelistInstance
     */
    remove(params: SafelistListInstanceRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SafelistInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SafelistInstance
     */
    fetch(callback?: (error: Error | null, item?: SafelistInstance) => any): Promise<SafelistInstance>;
    /**
     * Fetch a SafelistInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SafelistInstance
     */
    fetch(params: SafelistListInstanceFetchOptions, callback?: (error: Error | null, item?: SafelistInstance) => any): Promise<SafelistInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SafelistListInstance(version: V1): SafelistListInstance;
interface SafelistResource {
    sid: string;
    phone_number: string;
}
export declare class SafelistInstance {
    protected _version: V1;
    constructor(_version: V1, payload: SafelistResource);
    /**
     * The unique string that we created to identify the SafeList resource.
     */
    sid: string;
    /**
     * The phone number in SafeList.
     */
    phoneNumber: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        phoneNumber: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
