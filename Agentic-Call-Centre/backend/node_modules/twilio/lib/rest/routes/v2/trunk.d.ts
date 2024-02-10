/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../V2";
/**
 * Options to pass to update a TrunkInstance
 */
export interface TrunkContextUpdateOptions {
    /** The Inbound Processing Region used for this SIP Trunk for voice */
    voiceRegion?: string;
    /** A human readable description of this resource, up to 64 characters. */
    friendlyName?: string;
}
export interface TrunkContext {
    /**
     * Fetch a TrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    fetch(callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Update a TrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    update(callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Update a TrunkInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    update(params: TrunkContextUpdateOptions, callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface TrunkContextSolution {
    sipTrunkDomain: string;
}
export declare class TrunkContextImpl implements TrunkContext {
    protected _version: V2;
    protected _solution: TrunkContextSolution;
    protected _uri: string;
    constructor(_version: V2, sipTrunkDomain: string);
    fetch(callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    update(params?: TrunkContextUpdateOptions | ((error: Error | null, item?: TrunkInstance) => any), callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): TrunkContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface TrunkResource {
    sip_trunk_domain: string;
    url: string;
    sid: string;
    account_sid: string;
    friendly_name: string;
    voice_region: string;
    date_created: Date;
    date_updated: Date;
}
export declare class TrunkInstance {
    protected _version: V2;
    protected _solution: TrunkContextSolution;
    protected _context?: TrunkContext;
    constructor(_version: V2, payload: TrunkResource, sipTrunkDomain?: string);
    /**
     * The absolute URL of the SIP Trunk
     */
    sipTrunkDomain: string;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    /**
     * A 34 character string that uniquely identifies the Inbound Processing Region assignments for this SIP Trunk.
     */
    sid: string;
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * A human readable description of the Inbound Processing Region assignments for this SIP Trunk, up to 64 characters.
     */
    friendlyName: string;
    /**
     * The Inbound Processing Region used for this SIP Trunk for voice.
     */
    voiceRegion: string;
    /**
     * The date that this SIP Trunk was assigned an Inbound Processing Region, given in ISO 8601 format.
     */
    dateCreated: Date;
    /**
     * The date that the Inbound Processing Region was updated for this SIP Trunk, given in ISO 8601 format.
     */
    dateUpdated: Date;
    private get _proxy();
    /**
     * Fetch a TrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    fetch(callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Update a TrunkInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    update(callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Update a TrunkInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TrunkInstance
     */
    update(params: TrunkContextUpdateOptions, callback?: (error: Error | null, item?: TrunkInstance) => any): Promise<TrunkInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sipTrunkDomain: string;
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
export interface TrunkSolution {
}
export interface TrunkListInstance {
    _version: V2;
    _solution: TrunkSolution;
    _uri: string;
    (sipTrunkDomain: string): TrunkContext;
    get(sipTrunkDomain: string): TrunkContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TrunkListInstance(version: V2): TrunkListInstance;
export {};
