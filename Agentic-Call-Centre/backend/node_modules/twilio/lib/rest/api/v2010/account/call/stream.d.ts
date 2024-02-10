/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2010 from "../../../V2010";
export type StreamStatus = "in-progress" | "stopped";
export type StreamTrack = "inbound_track" | "outbound_track" | "both_tracks";
export type StreamUpdateStatus = "stopped";
/**
 * Options to pass to update a StreamInstance
 */
export interface StreamContextUpdateOptions {
    /**  */
    status: StreamUpdateStatus;
}
/**
 * Options to pass to create a StreamInstance
 */
export interface StreamListInstanceCreateOptions {
    /** Relative or absolute url where WebSocket connection will be established. */
    url: string;
    /** The user-specified name of this Stream, if one was given when the Stream was created. This may be used to stop the Stream. */
    name?: string;
    /**  */
    track?: StreamTrack;
    /** Absolute URL of the status callback. */
    statusCallback?: string;
    /** The http method for the status_callback (one of GET, POST). */
    statusCallbackMethod?: string;
    /** Parameter name */
    "parameter1.name"?: string;
    /** Parameter value */
    "parameter1.value"?: string;
    /** Parameter name */
    "parameter2.name"?: string;
    /** Parameter value */
    "parameter2.value"?: string;
    /** Parameter name */
    "parameter3.name"?: string;
    /** Parameter value */
    "parameter3.value"?: string;
    /** Parameter name */
    "parameter4.name"?: string;
    /** Parameter value */
    "parameter4.value"?: string;
    /** Parameter name */
    "parameter5.name"?: string;
    /** Parameter value */
    "parameter5.value"?: string;
    /** Parameter name */
    "parameter6.name"?: string;
    /** Parameter value */
    "parameter6.value"?: string;
    /** Parameter name */
    "parameter7.name"?: string;
    /** Parameter value */
    "parameter7.value"?: string;
    /** Parameter name */
    "parameter8.name"?: string;
    /** Parameter value */
    "parameter8.value"?: string;
    /** Parameter name */
    "parameter9.name"?: string;
    /** Parameter value */
    "parameter9.value"?: string;
    /** Parameter name */
    "parameter10.name"?: string;
    /** Parameter value */
    "parameter10.value"?: string;
    /** Parameter name */
    "parameter11.name"?: string;
    /** Parameter value */
    "parameter11.value"?: string;
    /** Parameter name */
    "parameter12.name"?: string;
    /** Parameter value */
    "parameter12.value"?: string;
    /** Parameter name */
    "parameter13.name"?: string;
    /** Parameter value */
    "parameter13.value"?: string;
    /** Parameter name */
    "parameter14.name"?: string;
    /** Parameter value */
    "parameter14.value"?: string;
    /** Parameter name */
    "parameter15.name"?: string;
    /** Parameter value */
    "parameter15.value"?: string;
    /** Parameter name */
    "parameter16.name"?: string;
    /** Parameter value */
    "parameter16.value"?: string;
    /** Parameter name */
    "parameter17.name"?: string;
    /** Parameter value */
    "parameter17.value"?: string;
    /** Parameter name */
    "parameter18.name"?: string;
    /** Parameter value */
    "parameter18.value"?: string;
    /** Parameter name */
    "parameter19.name"?: string;
    /** Parameter value */
    "parameter19.value"?: string;
    /** Parameter name */
    "parameter20.name"?: string;
    /** Parameter value */
    "parameter20.value"?: string;
    /** Parameter name */
    "parameter21.name"?: string;
    /** Parameter value */
    "parameter21.value"?: string;
    /** Parameter name */
    "parameter22.name"?: string;
    /** Parameter value */
    "parameter22.value"?: string;
    /** Parameter name */
    "parameter23.name"?: string;
    /** Parameter value */
    "parameter23.value"?: string;
    /** Parameter name */
    "parameter24.name"?: string;
    /** Parameter value */
    "parameter24.value"?: string;
    /** Parameter name */
    "parameter25.name"?: string;
    /** Parameter value */
    "parameter25.value"?: string;
    /** Parameter name */
    "parameter26.name"?: string;
    /** Parameter value */
    "parameter26.value"?: string;
    /** Parameter name */
    "parameter27.name"?: string;
    /** Parameter value */
    "parameter27.value"?: string;
    /** Parameter name */
    "parameter28.name"?: string;
    /** Parameter value */
    "parameter28.value"?: string;
    /** Parameter name */
    "parameter29.name"?: string;
    /** Parameter value */
    "parameter29.value"?: string;
    /** Parameter name */
    "parameter30.name"?: string;
    /** Parameter value */
    "parameter30.value"?: string;
    /** Parameter name */
    "parameter31.name"?: string;
    /** Parameter value */
    "parameter31.value"?: string;
    /** Parameter name */
    "parameter32.name"?: string;
    /** Parameter value */
    "parameter32.value"?: string;
    /** Parameter name */
    "parameter33.name"?: string;
    /** Parameter value */
    "parameter33.value"?: string;
    /** Parameter name */
    "parameter34.name"?: string;
    /** Parameter value */
    "parameter34.value"?: string;
    /** Parameter name */
    "parameter35.name"?: string;
    /** Parameter value */
    "parameter35.value"?: string;
    /** Parameter name */
    "parameter36.name"?: string;
    /** Parameter value */
    "parameter36.value"?: string;
    /** Parameter name */
    "parameter37.name"?: string;
    /** Parameter value */
    "parameter37.value"?: string;
    /** Parameter name */
    "parameter38.name"?: string;
    /** Parameter value */
    "parameter38.value"?: string;
    /** Parameter name */
    "parameter39.name"?: string;
    /** Parameter value */
    "parameter39.value"?: string;
    /** Parameter name */
    "parameter40.name"?: string;
    /** Parameter value */
    "parameter40.value"?: string;
    /** Parameter name */
    "parameter41.name"?: string;
    /** Parameter value */
    "parameter41.value"?: string;
    /** Parameter name */
    "parameter42.name"?: string;
    /** Parameter value */
    "parameter42.value"?: string;
    /** Parameter name */
    "parameter43.name"?: string;
    /** Parameter value */
    "parameter43.value"?: string;
    /** Parameter name */
    "parameter44.name"?: string;
    /** Parameter value */
    "parameter44.value"?: string;
    /** Parameter name */
    "parameter45.name"?: string;
    /** Parameter value */
    "parameter45.value"?: string;
    /** Parameter name */
    "parameter46.name"?: string;
    /** Parameter value */
    "parameter46.value"?: string;
    /** Parameter name */
    "parameter47.name"?: string;
    /** Parameter value */
    "parameter47.value"?: string;
    /** Parameter name */
    "parameter48.name"?: string;
    /** Parameter value */
    "parameter48.value"?: string;
    /** Parameter name */
    "parameter49.name"?: string;
    /** Parameter value */
    "parameter49.value"?: string;
    /** Parameter name */
    "parameter50.name"?: string;
    /** Parameter value */
    "parameter50.value"?: string;
    /** Parameter name */
    "parameter51.name"?: string;
    /** Parameter value */
    "parameter51.value"?: string;
    /** Parameter name */
    "parameter52.name"?: string;
    /** Parameter value */
    "parameter52.value"?: string;
    /** Parameter name */
    "parameter53.name"?: string;
    /** Parameter value */
    "parameter53.value"?: string;
    /** Parameter name */
    "parameter54.name"?: string;
    /** Parameter value */
    "parameter54.value"?: string;
    /** Parameter name */
    "parameter55.name"?: string;
    /** Parameter value */
    "parameter55.value"?: string;
    /** Parameter name */
    "parameter56.name"?: string;
    /** Parameter value */
    "parameter56.value"?: string;
    /** Parameter name */
    "parameter57.name"?: string;
    /** Parameter value */
    "parameter57.value"?: string;
    /** Parameter name */
    "parameter58.name"?: string;
    /** Parameter value */
    "parameter58.value"?: string;
    /** Parameter name */
    "parameter59.name"?: string;
    /** Parameter value */
    "parameter59.value"?: string;
    /** Parameter name */
    "parameter60.name"?: string;
    /** Parameter value */
    "parameter60.value"?: string;
    /** Parameter name */
    "parameter61.name"?: string;
    /** Parameter value */
    "parameter61.value"?: string;
    /** Parameter name */
    "parameter62.name"?: string;
    /** Parameter value */
    "parameter62.value"?: string;
    /** Parameter name */
    "parameter63.name"?: string;
    /** Parameter value */
    "parameter63.value"?: string;
    /** Parameter name */
    "parameter64.name"?: string;
    /** Parameter value */
    "parameter64.value"?: string;
    /** Parameter name */
    "parameter65.name"?: string;
    /** Parameter value */
    "parameter65.value"?: string;
    /** Parameter name */
    "parameter66.name"?: string;
    /** Parameter value */
    "parameter66.value"?: string;
    /** Parameter name */
    "parameter67.name"?: string;
    /** Parameter value */
    "parameter67.value"?: string;
    /** Parameter name */
    "parameter68.name"?: string;
    /** Parameter value */
    "parameter68.value"?: string;
    /** Parameter name */
    "parameter69.name"?: string;
    /** Parameter value */
    "parameter69.value"?: string;
    /** Parameter name */
    "parameter70.name"?: string;
    /** Parameter value */
    "parameter70.value"?: string;
    /** Parameter name */
    "parameter71.name"?: string;
    /** Parameter value */
    "parameter71.value"?: string;
    /** Parameter name */
    "parameter72.name"?: string;
    /** Parameter value */
    "parameter72.value"?: string;
    /** Parameter name */
    "parameter73.name"?: string;
    /** Parameter value */
    "parameter73.value"?: string;
    /** Parameter name */
    "parameter74.name"?: string;
    /** Parameter value */
    "parameter74.value"?: string;
    /** Parameter name */
    "parameter75.name"?: string;
    /** Parameter value */
    "parameter75.value"?: string;
    /** Parameter name */
    "parameter76.name"?: string;
    /** Parameter value */
    "parameter76.value"?: string;
    /** Parameter name */
    "parameter77.name"?: string;
    /** Parameter value */
    "parameter77.value"?: string;
    /** Parameter name */
    "parameter78.name"?: string;
    /** Parameter value */
    "parameter78.value"?: string;
    /** Parameter name */
    "parameter79.name"?: string;
    /** Parameter value */
    "parameter79.value"?: string;
    /** Parameter name */
    "parameter80.name"?: string;
    /** Parameter value */
    "parameter80.value"?: string;
    /** Parameter name */
    "parameter81.name"?: string;
    /** Parameter value */
    "parameter81.value"?: string;
    /** Parameter name */
    "parameter82.name"?: string;
    /** Parameter value */
    "parameter82.value"?: string;
    /** Parameter name */
    "parameter83.name"?: string;
    /** Parameter value */
    "parameter83.value"?: string;
    /** Parameter name */
    "parameter84.name"?: string;
    /** Parameter value */
    "parameter84.value"?: string;
    /** Parameter name */
    "parameter85.name"?: string;
    /** Parameter value */
    "parameter85.value"?: string;
    /** Parameter name */
    "parameter86.name"?: string;
    /** Parameter value */
    "parameter86.value"?: string;
    /** Parameter name */
    "parameter87.name"?: string;
    /** Parameter value */
    "parameter87.value"?: string;
    /** Parameter name */
    "parameter88.name"?: string;
    /** Parameter value */
    "parameter88.value"?: string;
    /** Parameter name */
    "parameter89.name"?: string;
    /** Parameter value */
    "parameter89.value"?: string;
    /** Parameter name */
    "parameter90.name"?: string;
    /** Parameter value */
    "parameter90.value"?: string;
    /** Parameter name */
    "parameter91.name"?: string;
    /** Parameter value */
    "parameter91.value"?: string;
    /** Parameter name */
    "parameter92.name"?: string;
    /** Parameter value */
    "parameter92.value"?: string;
    /** Parameter name */
    "parameter93.name"?: string;
    /** Parameter value */
    "parameter93.value"?: string;
    /** Parameter name */
    "parameter94.name"?: string;
    /** Parameter value */
    "parameter94.value"?: string;
    /** Parameter name */
    "parameter95.name"?: string;
    /** Parameter value */
    "parameter95.value"?: string;
    /** Parameter name */
    "parameter96.name"?: string;
    /** Parameter value */
    "parameter96.value"?: string;
    /** Parameter name */
    "parameter97.name"?: string;
    /** Parameter value */
    "parameter97.value"?: string;
    /** Parameter name */
    "parameter98.name"?: string;
    /** Parameter value */
    "parameter98.value"?: string;
    /** Parameter name */
    "parameter99.name"?: string;
    /** Parameter value */
    "parameter99.value"?: string;
}
export interface StreamContext {
    /**
     * Update a StreamInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed StreamInstance
     */
    update(params: StreamContextUpdateOptions, callback?: (error: Error | null, item?: StreamInstance) => any): Promise<StreamInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface StreamContextSolution {
    accountSid: string;
    callSid: string;
    sid: string;
}
export declare class StreamContextImpl implements StreamContext {
    protected _version: V2010;
    protected _solution: StreamContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, callSid: string, sid: string);
    update(params: StreamContextUpdateOptions, callback?: (error: Error | null, item?: StreamInstance) => any): Promise<StreamInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): StreamContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface StreamResource {
    sid: string;
    account_sid: string;
    call_sid: string;
    name: string;
    status: StreamStatus;
    date_updated: Date;
    uri: string;
}
export declare class StreamInstance {
    protected _version: V2010;
    protected _solution: StreamContextSolution;
    protected _context?: StreamContext;
    constructor(_version: V2010, payload: StreamResource, accountSid: string, callSid: string, sid?: string);
    /**
     * The SID of the Stream resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created this Stream resource.
     */
    accountSid: string;
    /**
     * The SID of the [Call](https://www.twilio.com/docs/voice/api/call-resource) the Stream resource is associated with.
     */
    callSid: string;
    /**
     * The user-specified name of this Stream, if one was given when the Stream was created. This may be used to stop the Stream.
     */
    name: string;
    status: StreamStatus;
    /**
     * The date and time in GMT that this resource was last updated, specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Update a StreamInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed StreamInstance
     */
    update(params: StreamContextUpdateOptions, callback?: (error: Error | null, item?: StreamInstance) => any): Promise<StreamInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        callSid: string;
        name: string;
        status: StreamStatus;
        dateUpdated: Date;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface StreamSolution {
    accountSid: string;
    callSid: string;
}
export interface StreamListInstance {
    _version: V2010;
    _solution: StreamSolution;
    _uri: string;
    (sid: string): StreamContext;
    get(sid: string): StreamContext;
    /**
     * Create a StreamInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed StreamInstance
     */
    create(params: StreamListInstanceCreateOptions, callback?: (error: Error | null, item?: StreamInstance) => any): Promise<StreamInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function StreamListInstance(version: V2010, accountSid: string, callSid: string): StreamListInstance;
export {};
