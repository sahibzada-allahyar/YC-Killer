/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to fetch a SettingInstance
 */
export interface SettingContextFetchOptions {
    /** The unique SID identifier of the Subaccount. */
    subaccountSid?: string;
}
/**
 * Options to pass to update a SettingInstance
 */
export interface SettingContextUpdateOptions {
    /** A boolean flag to enable Advanced Features for Voice Insights. */
    advancedFeatures?: boolean;
    /** A boolean flag to enable Voice Trace. */
    voiceTrace?: boolean;
    /** The unique SID identifier of the Subaccount. */
    subaccountSid?: string;
}
export interface SettingContext {
    /**
     * Fetch a SettingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingInstance
     */
    fetch(callback?: (error: Error | null, item?: SettingInstance) => any): Promise<SettingInstance>;
    /**
     * Fetch a SettingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingInstance
     */
    fetch(params: SettingContextFetchOptions, callback?: (error: Error | null, item?: SettingInstance) => any): Promise<SettingInstance>;
    /**
     * Update a SettingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingInstance
     */
    update(callback?: (error: Error | null, item?: SettingInstance) => any): Promise<SettingInstance>;
    /**
     * Update a SettingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingInstance
     */
    update(params: SettingContextUpdateOptions, callback?: (error: Error | null, item?: SettingInstance) => any): Promise<SettingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SettingContextSolution {
}
export declare class SettingContextImpl implements SettingContext {
    protected _version: V1;
    protected _solution: SettingContextSolution;
    protected _uri: string;
    constructor(_version: V1);
    fetch(params?: SettingContextFetchOptions | ((error: Error | null, item?: SettingInstance) => any), callback?: (error: Error | null, item?: SettingInstance) => any): Promise<SettingInstance>;
    update(params?: SettingContextUpdateOptions | ((error: Error | null, item?: SettingInstance) => any), callback?: (error: Error | null, item?: SettingInstance) => any): Promise<SettingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SettingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SettingResource {
    account_sid: string;
    advanced_features: boolean;
    voice_trace: boolean;
    url: string;
}
export declare class SettingInstance {
    protected _version: V1;
    protected _solution: SettingContextSolution;
    protected _context?: SettingContext;
    constructor(_version: V1, payload: SettingResource);
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * A boolean flag indicating whether Advanced Features for Voice Insights are enabled.
     */
    advancedFeatures: boolean;
    /**
     * A boolean flag indicating whether Voice Trace is enabled.
     */
    voiceTrace: boolean;
    /**
     * The URL of this resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a SettingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingInstance
     */
    fetch(callback?: (error: Error | null, item?: SettingInstance) => any): Promise<SettingInstance>;
    /**
     * Fetch a SettingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingInstance
     */
    fetch(params: SettingContextFetchOptions, callback?: (error: Error | null, item?: SettingInstance) => any): Promise<SettingInstance>;
    /**
     * Update a SettingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingInstance
     */
    update(callback?: (error: Error | null, item?: SettingInstance) => any): Promise<SettingInstance>;
    /**
     * Update a SettingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingInstance
     */
    update(params: SettingContextUpdateOptions, callback?: (error: Error | null, item?: SettingInstance) => any): Promise<SettingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        advancedFeatures: boolean;
        voiceTrace: boolean;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SettingSolution {
}
export interface SettingListInstance {
    _version: V1;
    _solution: SettingSolution;
    _uri: string;
    (): SettingContext;
    get(): SettingContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SettingListInstance(version: V1): SettingListInstance;
export {};
