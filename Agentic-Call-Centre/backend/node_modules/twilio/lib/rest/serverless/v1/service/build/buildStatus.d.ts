/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
export type BuildStatusStatus = "building" | "completed" | "failed";
export interface BuildStatusContext {
    /**
     * Fetch a BuildStatusInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BuildStatusInstance
     */
    fetch(callback?: (error: Error | null, item?: BuildStatusInstance) => any): Promise<BuildStatusInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface BuildStatusContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class BuildStatusContextImpl implements BuildStatusContext {
    protected _version: V1;
    protected _solution: BuildStatusContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: BuildStatusInstance) => any): Promise<BuildStatusInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): BuildStatusContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface BuildStatusResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    status: BuildStatusStatus;
    url: string;
}
export declare class BuildStatusInstance {
    protected _version: V1;
    protected _solution: BuildStatusContextSolution;
    protected _context?: BuildStatusContext;
    constructor(_version: V1, payload: BuildStatusResource, serviceSid: string, sid: string);
    /**
     * The unique string that we created to identify the Build resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Build resource.
     */
    accountSid: string;
    /**
     * The SID of the Service that the Build resource is associated with.
     */
    serviceSid: string;
    status: BuildStatusStatus;
    /**
     * The absolute URL of the Build Status resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a BuildStatusInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BuildStatusInstance
     */
    fetch(callback?: (error: Error | null, item?: BuildStatusInstance) => any): Promise<BuildStatusInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        status: BuildStatusStatus;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface BuildStatusSolution {
    serviceSid: string;
    sid: string;
}
export interface BuildStatusListInstance {
    _version: V1;
    _solution: BuildStatusSolution;
    _uri: string;
    (): BuildStatusContext;
    get(): BuildStatusContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BuildStatusListInstance(version: V1, serviceSid: string, sid: string): BuildStatusListInstance;
export {};
