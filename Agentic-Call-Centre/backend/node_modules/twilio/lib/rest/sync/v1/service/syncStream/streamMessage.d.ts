/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
/**
 * Options to pass to create a StreamMessageInstance
 */
export interface StreamMessageListInstanceCreateOptions {
    /** A JSON string that represents an arbitrary, schema-less object that makes up the Stream Message body. Can be up to 4 KiB in length. */
    data: any;
}
export interface StreamMessageSolution {
    serviceSid: string;
    streamSid: string;
}
export interface StreamMessageListInstance {
    _version: V1;
    _solution: StreamMessageSolution;
    _uri: string;
    /**
     * Create a StreamMessageInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed StreamMessageInstance
     */
    create(params: StreamMessageListInstanceCreateOptions, callback?: (error: Error | null, item?: StreamMessageInstance) => any): Promise<StreamMessageInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function StreamMessageListInstance(version: V1, serviceSid: string, streamSid: string): StreamMessageListInstance;
interface StreamMessageResource {
    sid: string;
    data: any;
}
export declare class StreamMessageInstance {
    protected _version: V1;
    constructor(_version: V1, payload: StreamMessageResource, serviceSid: string, streamSid: string);
    /**
     * The unique string that we created to identify the Stream Message resource.
     */
    sid: string;
    /**
     * An arbitrary, schema-less object that contains the Stream Message body. Can be up to 4 KiB in length.
     */
    data: any;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        data: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
