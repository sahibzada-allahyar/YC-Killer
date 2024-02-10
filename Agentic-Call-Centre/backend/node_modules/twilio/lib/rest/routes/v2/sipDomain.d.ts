/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../V2";
/**
 * Options to pass to update a SipDomainInstance
 */
export interface SipDomainContextUpdateOptions {
    /**  */
    voiceRegion?: string;
    /**  */
    friendlyName?: string;
}
export interface SipDomainContext {
    /**
     * Fetch a SipDomainInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SipDomainInstance
     */
    fetch(callback?: (error: Error | null, item?: SipDomainInstance) => any): Promise<SipDomainInstance>;
    /**
     * Update a SipDomainInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SipDomainInstance
     */
    update(callback?: (error: Error | null, item?: SipDomainInstance) => any): Promise<SipDomainInstance>;
    /**
     * Update a SipDomainInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SipDomainInstance
     */
    update(params: SipDomainContextUpdateOptions, callback?: (error: Error | null, item?: SipDomainInstance) => any): Promise<SipDomainInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SipDomainContextSolution {
    sipDomain: string;
}
export declare class SipDomainContextImpl implements SipDomainContext {
    protected _version: V2;
    protected _solution: SipDomainContextSolution;
    protected _uri: string;
    constructor(_version: V2, sipDomain: string);
    fetch(callback?: (error: Error | null, item?: SipDomainInstance) => any): Promise<SipDomainInstance>;
    update(params?: SipDomainContextUpdateOptions | ((error: Error | null, item?: SipDomainInstance) => any), callback?: (error: Error | null, item?: SipDomainInstance) => any): Promise<SipDomainInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SipDomainContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SipDomainResource {
    sip_domain: string;
    url: string;
    sid: string;
    account_sid: string;
    friendly_name: string;
    voice_region: string;
    date_created: Date;
    date_updated: Date;
}
export declare class SipDomainInstance {
    protected _version: V2;
    protected _solution: SipDomainContextSolution;
    protected _context?: SipDomainContext;
    constructor(_version: V2, payload: SipDomainResource, sipDomain?: string);
    sipDomain: string;
    url: string;
    sid: string;
    accountSid: string;
    friendlyName: string;
    voiceRegion: string;
    dateCreated: Date;
    dateUpdated: Date;
    private get _proxy();
    /**
     * Fetch a SipDomainInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SipDomainInstance
     */
    fetch(callback?: (error: Error | null, item?: SipDomainInstance) => any): Promise<SipDomainInstance>;
    /**
     * Update a SipDomainInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SipDomainInstance
     */
    update(callback?: (error: Error | null, item?: SipDomainInstance) => any): Promise<SipDomainInstance>;
    /**
     * Update a SipDomainInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SipDomainInstance
     */
    update(params: SipDomainContextUpdateOptions, callback?: (error: Error | null, item?: SipDomainInstance) => any): Promise<SipDomainInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sipDomain: string;
        url: string;
        sid: string;
        accountSid: string;
        friendlyName: string;
        voiceRegion: string;
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SipDomainSolution {
}
export interface SipDomainListInstance {
    _version: V2;
    _solution: SipDomainSolution;
    _uri: string;
    (sipDomain: string): SipDomainContext;
    get(sipDomain: string): SipDomainContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SipDomainListInstance(version: V2): SipDomainListInstance;
export {};
