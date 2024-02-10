/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Wireless from "../../Wireless";
/**
 * Options to pass to fetch a UsageInstance
 */
export interface UsageContextFetchOptions {
    /**  */
    end?: string;
    /**  */
    start?: string;
}
export interface UsageContext {
    /**
     * Fetch a UsageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UsageInstance
     */
    fetch(callback?: (error: Error | null, item?: UsageInstance) => any): Promise<UsageInstance>;
    /**
     * Fetch a UsageInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UsageInstance
     */
    fetch(params: UsageContextFetchOptions, callback?: (error: Error | null, item?: UsageInstance) => any): Promise<UsageInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface UsageContextSolution {
    simSid: string;
}
export declare class UsageContextImpl implements UsageContext {
    protected _version: Wireless;
    protected _solution: UsageContextSolution;
    protected _uri: string;
    constructor(_version: Wireless, simSid: string);
    fetch(params?: UsageContextFetchOptions | ((error: Error | null, item?: UsageInstance) => any), callback?: (error: Error | null, item?: UsageInstance) => any): Promise<UsageInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): UsageContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface UsageResource {
    sim_sid: string;
    sim_unique_name: string;
    account_sid: string;
    period: any;
    commands_usage: any;
    commands_costs: any;
    data_usage: any;
    data_costs: any;
    url: string;
}
export declare class UsageInstance {
    protected _version: Wireless;
    protected _solution: UsageContextSolution;
    protected _context?: UsageContext;
    constructor(_version: Wireless, payload: UsageResource, simSid: string);
    simSid: string;
    simUniqueName: string;
    accountSid: string;
    period: any;
    commandsUsage: any;
    commandsCosts: any;
    dataUsage: any;
    dataCosts: any;
    url: string;
    private get _proxy();
    /**
     * Fetch a UsageInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UsageInstance
     */
    fetch(callback?: (error: Error | null, item?: UsageInstance) => any): Promise<UsageInstance>;
    /**
     * Fetch a UsageInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UsageInstance
     */
    fetch(params: UsageContextFetchOptions, callback?: (error: Error | null, item?: UsageInstance) => any): Promise<UsageInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        simSid: string;
        simUniqueName: string;
        accountSid: string;
        period: any;
        commandsUsage: any;
        commandsCosts: any;
        dataUsage: any;
        dataCosts: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface UsageSolution {
    simSid: string;
}
export interface UsageListInstance {
    _version: Wireless;
    _solution: UsageSolution;
    _uri: string;
    (): UsageContext;
    get(): UsageContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function UsageListInstance(version: Wireless, simSid: string): UsageListInstance;
export {};
