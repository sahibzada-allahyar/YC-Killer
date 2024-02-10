/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
export interface ApprovalFetchContext {
    /**
     * Fetch a ApprovalFetchInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ApprovalFetchInstance
     */
    fetch(callback?: (error: Error | null, item?: ApprovalFetchInstance) => any): Promise<ApprovalFetchInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ApprovalFetchContextSolution {
    sid: string;
}
export declare class ApprovalFetchContextImpl implements ApprovalFetchContext {
    protected _version: V1;
    protected _solution: ApprovalFetchContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: ApprovalFetchInstance) => any): Promise<ApprovalFetchInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ApprovalFetchContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ApprovalFetchResource {
    sid: string;
    account_sid: string;
    whatsapp: any;
    url: string;
}
export declare class ApprovalFetchInstance {
    protected _version: V1;
    protected _solution: ApprovalFetchContextSolution;
    protected _context?: ApprovalFetchContext;
    constructor(_version: V1, payload: ApprovalFetchResource, sid: string);
    /**
     * The unique string that that we created to identify the Content resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/usage/api/account) that created Content resource.
     */
    accountSid: string;
    /**
     * Contains the whatsapp approval information for the Content resource, with fields such as approval status, rejection reason, and category, amongst others.
     */
    whatsapp: any;
    /**
     * The URL of the resource, relative to `https://content.twilio.com`.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a ApprovalFetchInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ApprovalFetchInstance
     */
    fetch(callback?: (error: Error | null, item?: ApprovalFetchInstance) => any): Promise<ApprovalFetchInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        whatsapp: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ApprovalFetchSolution {
    sid: string;
}
export interface ApprovalFetchListInstance {
    _version: V1;
    _solution: ApprovalFetchSolution;
    _uri: string;
    (): ApprovalFetchContext;
    get(): ApprovalFetchContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ApprovalFetchListInstance(version: V1, sid: string): ApprovalFetchListInstance;
export {};
