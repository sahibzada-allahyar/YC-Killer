/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../../V1";
export interface FunctionVersionContentContext {
    /**
     * Fetch a FunctionVersionContentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FunctionVersionContentInstance
     */
    fetch(callback?: (error: Error | null, item?: FunctionVersionContentInstance) => any): Promise<FunctionVersionContentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface FunctionVersionContentContextSolution {
    serviceSid: string;
    functionSid: string;
    sid: string;
}
export declare class FunctionVersionContentContextImpl implements FunctionVersionContentContext {
    protected _version: V1;
    protected _solution: FunctionVersionContentContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, functionSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: FunctionVersionContentInstance) => any): Promise<FunctionVersionContentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): FunctionVersionContentContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface FunctionVersionContentResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    function_sid: string;
    content: string;
    url: string;
}
export declare class FunctionVersionContentInstance {
    protected _version: V1;
    protected _solution: FunctionVersionContentContextSolution;
    protected _context?: FunctionVersionContentContext;
    constructor(_version: V1, payload: FunctionVersionContentResource, serviceSid: string, functionSid: string, sid: string);
    /**
     * The unique string that we created to identify the Function Version resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Function Version resource.
     */
    accountSid: string;
    /**
     * The SID of the Service that the Function Version resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the Function that is the parent of the Function Version.
     */
    functionSid: string;
    /**
     * The content of the Function Version resource.
     */
    content: string;
    url: string;
    private get _proxy();
    /**
     * Fetch a FunctionVersionContentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FunctionVersionContentInstance
     */
    fetch(callback?: (error: Error | null, item?: FunctionVersionContentInstance) => any): Promise<FunctionVersionContentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        functionSid: string;
        content: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface FunctionVersionContentSolution {
    serviceSid: string;
    functionSid: string;
    sid: string;
}
export interface FunctionVersionContentListInstance {
    _version: V1;
    _solution: FunctionVersionContentSolution;
    _uri: string;
    (): FunctionVersionContentContext;
    get(): FunctionVersionContentContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function FunctionVersionContentListInstance(version: V1, serviceSid: string, functionSid: string, sid: string): FunctionVersionContentListInstance;
export {};
