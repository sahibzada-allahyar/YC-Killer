/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export type ProvisioningStatusStatus = "active" | "in-progress" | "not-configured" | "failed";
export interface ProvisioningStatusContext {
    /**
     * Fetch a ProvisioningStatusInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ProvisioningStatusInstance
     */
    fetch(callback?: (error: Error | null, item?: ProvisioningStatusInstance) => any): Promise<ProvisioningStatusInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ProvisioningStatusContextSolution {
}
export declare class ProvisioningStatusContextImpl implements ProvisioningStatusContext {
    protected _version: V1;
    protected _solution: ProvisioningStatusContextSolution;
    protected _uri: string;
    constructor(_version: V1);
    fetch(callback?: (error: Error | null, item?: ProvisioningStatusInstance) => any): Promise<ProvisioningStatusInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ProvisioningStatusContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ProvisioningStatusResource {
    status: ProvisioningStatusStatus;
    url: string;
}
export declare class ProvisioningStatusInstance {
    protected _version: V1;
    protected _solution: ProvisioningStatusContextSolution;
    protected _context?: ProvisioningStatusContext;
    constructor(_version: V1, payload: ProvisioningStatusResource);
    status: ProvisioningStatusStatus;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a ProvisioningStatusInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ProvisioningStatusInstance
     */
    fetch(callback?: (error: Error | null, item?: ProvisioningStatusInstance) => any): Promise<ProvisioningStatusInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        status: ProvisioningStatusStatus;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ProvisioningStatusSolution {
}
export interface ProvisioningStatusListInstance {
    _version: V1;
    _solution: ProvisioningStatusSolution;
    _uri: string;
    (): ProvisioningStatusContext;
    get(): ProvisioningStatusContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ProvisioningStatusListInstance(version: V1): ProvisioningStatusListInstance;
export {};
