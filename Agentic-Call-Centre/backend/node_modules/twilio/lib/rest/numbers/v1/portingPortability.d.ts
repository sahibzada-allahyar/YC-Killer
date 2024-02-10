/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export type PortingPortabilityNumberType = "LOCAL" | "UNKNOWN" | "MOBILE" | "TOLL-FREE";
/**
 * Options to pass to fetch a PortingPortabilityInstance
 */
export interface PortingPortabilityContextFetchOptions {
    /** The SID of the account where the phone number(s) will be ported. */
    targetAccountSid?: string;
}
export interface PortingPortabilityContext {
    /**
     * Fetch a PortingPortabilityInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PortingPortabilityInstance
     */
    fetch(callback?: (error: Error | null, item?: PortingPortabilityInstance) => any): Promise<PortingPortabilityInstance>;
    /**
     * Fetch a PortingPortabilityInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PortingPortabilityInstance
     */
    fetch(params: PortingPortabilityContextFetchOptions, callback?: (error: Error | null, item?: PortingPortabilityInstance) => any): Promise<PortingPortabilityInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PortingPortabilityContextSolution {
    phoneNumber: string;
}
export declare class PortingPortabilityContextImpl implements PortingPortabilityContext {
    protected _version: V1;
    protected _solution: PortingPortabilityContextSolution;
    protected _uri: string;
    constructor(_version: V1, phoneNumber: string);
    fetch(params?: PortingPortabilityContextFetchOptions | ((error: Error | null, item?: PortingPortabilityInstance) => any), callback?: (error: Error | null, item?: PortingPortabilityInstance) => any): Promise<PortingPortabilityInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PortingPortabilityContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PortingPortabilityResource {
    phone_number: string;
    account_sid: string;
    portable: boolean;
    pin_and_account_number_required: boolean;
    not_portable_reason: string;
    not_portable_reason_code: number;
    number_type: PortingPortabilityNumberType;
    country: string;
    messaging_carrier: string;
    voice_carrier: string;
    url: string;
}
export declare class PortingPortabilityInstance {
    protected _version: V1;
    protected _solution: PortingPortabilityContextSolution;
    protected _context?: PortingPortabilityContext;
    constructor(_version: V1, payload: PortingPortabilityResource, phoneNumber?: string);
    /**
     * The phone number which portability is to be checked. Phone numbers are in E.164 format (e.g. +16175551212).
     */
    phoneNumber: string;
    /**
     * The target account sid to which the number will be ported
     */
    accountSid: string;
    /**
     * Boolean flag specifying if phone number is portable or not.
     */
    portable: boolean;
    /**
     * Boolean flag specifying if PIN and account number is required for the phone number.
     */
    pinAndAccountNumberRequired: boolean;
    /**
     * Reason why the phone number cannot be ported into Twilio, `null` otherwise.
     */
    notPortableReason: string;
    /**
     * The Portability Reason Code for the phone number if it cannot be ported into Twilio, `null` otherwise. One of `22131`, `22132`, `22130`, `22133`, `22102` or `22135`.
     */
    notPortableReasonCode: number;
    numberType: PortingPortabilityNumberType;
    /**
     * Country the phone number belongs to.
     */
    country: string;
    /**
     * Current messaging carrier of the phone number
     */
    messagingCarrier: string;
    /**
     * Current voice carrier of the phone number
     */
    voiceCarrier: string;
    /**
     * This is the url of the request that you\'re trying to reach out to locate the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a PortingPortabilityInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PortingPortabilityInstance
     */
    fetch(callback?: (error: Error | null, item?: PortingPortabilityInstance) => any): Promise<PortingPortabilityInstance>;
    /**
     * Fetch a PortingPortabilityInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PortingPortabilityInstance
     */
    fetch(params: PortingPortabilityContextFetchOptions, callback?: (error: Error | null, item?: PortingPortabilityInstance) => any): Promise<PortingPortabilityInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        phoneNumber: string;
        accountSid: string;
        portable: boolean;
        pinAndAccountNumberRequired: boolean;
        notPortableReason: string;
        notPortableReasonCode: number;
        numberType: PortingPortabilityNumberType;
        country: string;
        messagingCarrier: string;
        voiceCarrier: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PortingPortabilitySolution {
}
export interface PortingPortabilityListInstance {
    _version: V1;
    _solution: PortingPortabilitySolution;
    _uri: string;
    (phoneNumber: string): PortingPortabilityContext;
    get(phoneNumber: string): PortingPortabilityContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PortingPortabilityListInstance(version: V1): PortingPortabilityListInstance;
export {};
