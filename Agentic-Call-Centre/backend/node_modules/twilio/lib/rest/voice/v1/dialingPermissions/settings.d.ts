/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
/**
 * Options to pass to update a SettingsInstance
 */
export interface SettingsContextUpdateOptions {
    /** `true` for the sub-account to inherit voice dialing permissions from the Master Project; otherwise `false`. */
    dialingPermissionsInheritance?: boolean;
}
export interface SettingsContext {
    /**
     * Fetch a SettingsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingsInstance
     */
    fetch(callback?: (error: Error | null, item?: SettingsInstance) => any): Promise<SettingsInstance>;
    /**
     * Update a SettingsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingsInstance
     */
    update(callback?: (error: Error | null, item?: SettingsInstance) => any): Promise<SettingsInstance>;
    /**
     * Update a SettingsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingsInstance
     */
    update(params: SettingsContextUpdateOptions, callback?: (error: Error | null, item?: SettingsInstance) => any): Promise<SettingsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SettingsContextSolution {
}
export declare class SettingsContextImpl implements SettingsContext {
    protected _version: V1;
    protected _solution: SettingsContextSolution;
    protected _uri: string;
    constructor(_version: V1);
    fetch(callback?: (error: Error | null, item?: SettingsInstance) => any): Promise<SettingsInstance>;
    update(params?: SettingsContextUpdateOptions | ((error: Error | null, item?: SettingsInstance) => any), callback?: (error: Error | null, item?: SettingsInstance) => any): Promise<SettingsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SettingsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SettingsResource {
    dialing_permissions_inheritance: boolean;
    url: string;
}
export declare class SettingsInstance {
    protected _version: V1;
    protected _solution: SettingsContextSolution;
    protected _context?: SettingsContext;
    constructor(_version: V1, payload: SettingsResource);
    /**
     * `true` if the sub-account will inherit voice dialing permissions from the Master Project; otherwise `false`.
     */
    dialingPermissionsInheritance: boolean;
    /**
     * The absolute URL of this resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a SettingsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingsInstance
     */
    fetch(callback?: (error: Error | null, item?: SettingsInstance) => any): Promise<SettingsInstance>;
    /**
     * Update a SettingsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingsInstance
     */
    update(callback?: (error: Error | null, item?: SettingsInstance) => any): Promise<SettingsInstance>;
    /**
     * Update a SettingsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SettingsInstance
     */
    update(params: SettingsContextUpdateOptions, callback?: (error: Error | null, item?: SettingsInstance) => any): Promise<SettingsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        dialingPermissionsInheritance: boolean;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SettingsSolution {
}
export interface SettingsListInstance {
    _version: V1;
    _solution: SettingsSolution;
    _uri: string;
    (): SettingsContext;
    get(): SettingsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SettingsListInstance(version: V1): SettingsListInstance;
export {};
