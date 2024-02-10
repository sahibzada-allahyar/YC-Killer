/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
export declare class VideoV1RoomRoomRecordingRuleRules {
    "type"?: string;
    "all"?: boolean;
    "publisher"?: string;
    "track"?: string;
    "kind"?: string;
}
/**
 * Options to pass to update a RecordingRulesInstance
 */
export interface RecordingRulesListInstanceUpdateOptions {
    /** A JSON-encoded array of recording rules. */
    rules?: any;
}
export interface RecordingRulesSolution {
    roomSid: string;
}
export interface RecordingRulesListInstance {
    _version: V1;
    _solution: RecordingRulesSolution;
    _uri: string;
    /**
     * Fetch a RecordingRulesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingRulesInstance
     */
    fetch(callback?: (error: Error | null, item?: RecordingRulesInstance) => any): Promise<RecordingRulesInstance>;
    /**
     * Update a RecordingRulesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingRulesInstance
     */
    update(callback?: (error: Error | null, item?: RecordingRulesInstance) => any): Promise<RecordingRulesInstance>;
    /**
     * Update a RecordingRulesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingRulesInstance
     */
    update(params: RecordingRulesListInstanceUpdateOptions, callback?: (error: Error | null, item?: RecordingRulesInstance) => any): Promise<RecordingRulesInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RecordingRulesListInstance(version: V1, roomSid: string): RecordingRulesListInstance;
interface RecordingRulesResource {
    room_sid: string;
    rules: Array<VideoV1RoomRoomRecordingRuleRules>;
    date_created: Date;
    date_updated: Date;
}
export declare class RecordingRulesInstance {
    protected _version: V1;
    constructor(_version: V1, payload: RecordingRulesResource, roomSid: string);
    /**
     * The SID of the Room resource for the Recording Rules
     */
    roomSid: string;
    /**
     * A collection of Recording Rules that describe how to include or exclude matching tracks for recording
     */
    rules: Array<VideoV1RoomRoomRecordingRuleRules>;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        roomSid: string;
        rules: VideoV1RoomRoomRecordingRuleRules[];
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
