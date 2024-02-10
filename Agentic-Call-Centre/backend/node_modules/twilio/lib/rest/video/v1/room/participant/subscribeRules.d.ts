/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
export declare class VideoV1RoomRoomParticipantRoomParticipantSubscribeRuleRules {
    "type"?: string;
    "all"?: boolean;
    "publisher"?: string;
    "track"?: string;
    "kind"?: string;
    "priority"?: string;
}
/**
 * Options to pass to update a SubscribeRulesInstance
 */
export interface SubscribeRulesListInstanceUpdateOptions {
    /** A JSON-encoded array of subscribe rules. See the [Specifying Subscribe Rules](https://www.twilio.com/docs/video/api/track-subscriptions#specifying-sr) section for further information. */
    rules?: any;
}
export interface SubscribeRulesSolution {
    roomSid: string;
    participantSid: string;
}
export interface SubscribeRulesListInstance {
    _version: V1;
    _solution: SubscribeRulesSolution;
    _uri: string;
    /**
     * Fetch a SubscribeRulesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscribeRulesInstance
     */
    fetch(callback?: (error: Error | null, item?: SubscribeRulesInstance) => any): Promise<SubscribeRulesInstance>;
    /**
     * Update a SubscribeRulesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscribeRulesInstance
     */
    update(callback?: (error: Error | null, item?: SubscribeRulesInstance) => any): Promise<SubscribeRulesInstance>;
    /**
     * Update a SubscribeRulesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SubscribeRulesInstance
     */
    update(params: SubscribeRulesListInstanceUpdateOptions, callback?: (error: Error | null, item?: SubscribeRulesInstance) => any): Promise<SubscribeRulesInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SubscribeRulesListInstance(version: V1, roomSid: string, participantSid: string): SubscribeRulesListInstance;
interface SubscribeRulesResource {
    participant_sid: string;
    room_sid: string;
    rules: Array<VideoV1RoomRoomParticipantRoomParticipantSubscribeRuleRules>;
    date_created: Date;
    date_updated: Date;
}
export declare class SubscribeRulesInstance {
    protected _version: V1;
    constructor(_version: V1, payload: SubscribeRulesResource, roomSid: string, participantSid: string);
    /**
     * The SID of the Participant resource for the Subscribe Rules.
     */
    participantSid: string;
    /**
     * The SID of the Room resource for the Subscribe Rules
     */
    roomSid: string;
    /**
     * A collection of Subscribe Rules that describe how to include or exclude matching tracks. See the [Specifying Subscribe Rules](https://www.twilio.com/docs/video/api/track-subscriptions#specifying-sr) section for further information.
     */
    rules: Array<VideoV1RoomRoomParticipantRoomParticipantSubscribeRuleRules>;
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
        participantSid: string;
        roomSid: string;
        rules: VideoV1RoomRoomParticipantRoomParticipantSubscribeRuleRules[];
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
