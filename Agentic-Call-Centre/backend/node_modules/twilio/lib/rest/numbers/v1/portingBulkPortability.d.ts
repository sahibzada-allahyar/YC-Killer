/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export type PortingBulkPortabilityStatus = "in-progress" | "completed" | "expired";
/**
 * Options to pass to create a PortingBulkPortabilityInstance
 */
export interface PortingBulkPortabilityListInstanceCreateOptions {
    /** The phone numbers which portability is to be checked. This should be a list of strings. Phone numbers are in E.164 format (e.g. +16175551212). . */
    phoneNumbers: Array<string>;
}
export interface PortingBulkPortabilityContext {
    /**
     * Fetch a PortingBulkPortabilityInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PortingBulkPortabilityInstance
     */
    fetch(callback?: (error: Error | null, item?: PortingBulkPortabilityInstance) => any): Promise<PortingBulkPortabilityInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PortingBulkPortabilityContextSolution {
    sid: string;
}
export declare class PortingBulkPortabilityContextImpl implements PortingBulkPortabilityContext {
    protected _version: V1;
    protected _solution: PortingBulkPortabilityContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: PortingBulkPortabilityInstance) => any): Promise<PortingBulkPortabilityInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PortingBulkPortabilityContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PortingBulkPortabilityResource {
    sid: string;
    status: PortingBulkPortabilityStatus;
    datetime_created: Date;
    phone_numbers: Array<any>;
    url: string;
}
export declare class PortingBulkPortabilityInstance {
    protected _version: V1;
    protected _solution: PortingBulkPortabilityContextSolution;
    protected _context?: PortingBulkPortabilityContext;
    constructor(_version: V1, payload: PortingBulkPortabilityResource, sid?: string);
    /**
     * A 34 character string that uniquely identifies this Portability check.
     */
    sid: string;
    status: PortingBulkPortabilityStatus;
    /**
     * The date that the Portability check was created, given in ISO 8601 format.
     */
    datetimeCreated: Date;
    /**
     * Contains a list with all the information of the requested phone numbers. Each phone number contains the following properties: `phone_number`: The phone number which portability is to be checked. `portable`: Boolean flag specifying if phone number is portable or not. `not_portable_reason`: Reason why the phone number cannot be ported into Twilio, `null` otherwise. `not_portable_reason_code`: The Portability Reason Code for the phone number if it cannot be ported in Twilio, `null` otherwise. `pin_and_account_number_required`: Boolean flag specifying if PIN and account number is required for the phone number. `number_type`: The type of the requested phone number. `country` Country the phone number belongs to. `messaging_carrier` Current messaging carrier of the phone number. `voice_carrier` Current voice carrier of the phone number.
     */
    phoneNumbers: Array<any>;
    /**
     * This is the url of the request that you\'re trying to reach out to locate the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a PortingBulkPortabilityInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PortingBulkPortabilityInstance
     */
    fetch(callback?: (error: Error | null, item?: PortingBulkPortabilityInstance) => any): Promise<PortingBulkPortabilityInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        status: PortingBulkPortabilityStatus;
        datetimeCreated: Date;
        phoneNumbers: any[];
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PortingBulkPortabilitySolution {
}
export interface PortingBulkPortabilityListInstance {
    _version: V1;
    _solution: PortingBulkPortabilitySolution;
    _uri: string;
    (sid: string): PortingBulkPortabilityContext;
    get(sid: string): PortingBulkPortabilityContext;
    /**
     * Create a PortingBulkPortabilityInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PortingBulkPortabilityInstance
     */
    create(params: PortingBulkPortabilityListInstanceCreateOptions, callback?: (error: Error | null, item?: PortingBulkPortabilityInstance) => any): Promise<PortingBulkPortabilityInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PortingBulkPortabilityListInstance(version: V1): PortingBulkPortabilityListInstance;
export {};
