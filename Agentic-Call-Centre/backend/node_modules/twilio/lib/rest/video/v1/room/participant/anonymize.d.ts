/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../../V1";
export type AnonymizeStatus = "connected" | "disconnected";
export interface AnonymizeContext {
    /**
     * Update a AnonymizeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AnonymizeInstance
     */
    update(callback?: (error: Error | null, item?: AnonymizeInstance) => any): Promise<AnonymizeInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AnonymizeContextSolution {
    roomSid: string;
    sid: string;
}
export declare class AnonymizeContextImpl implements AnonymizeContext {
    protected _version: V1;
    protected _solution: AnonymizeContextSolution;
    protected _uri: string;
    constructor(_version: V1, roomSid: string, sid: string);
    update(callback?: (error: Error | null, item?: AnonymizeInstance) => any): Promise<AnonymizeInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AnonymizeContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AnonymizeResource {
    sid: string;
    room_sid: string;
    account_sid: string;
    status: AnonymizeStatus;
    identity: string;
    date_created: Date;
    date_updated: Date;
    start_time: Date;
    end_time: Date;
    duration: number;
    url: string;
}
export declare class AnonymizeInstance {
    protected _version: V1;
    protected _solution: AnonymizeContextSolution;
    protected _context?: AnonymizeContext;
    constructor(_version: V1, payload: AnonymizeResource, roomSid: string, sid: string);
    /**
     * The unique string that we created to identify the RoomParticipant resource.
     */
    sid: string;
    /**
     * The SID of the participant\'s room.
     */
    roomSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the RoomParticipant resource.
     */
    accountSid: string;
    status: AnonymizeStatus;
    /**
     * The SID of the participant.
     */
    identity: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The time of participant connected to the room in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#UTC) format.
     */
    startTime: Date;
    /**
     * The time when the participant disconnected from the room in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#UTC) format.
     */
    endTime: Date;
    /**
     * The duration in seconds that the participant was `connected`. Populated only after the participant is `disconnected`.
     */
    duration: number;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Update a AnonymizeInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AnonymizeInstance
     */
    update(callback?: (error: Error | null, item?: AnonymizeInstance) => any): Promise<AnonymizeInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        roomSid: string;
        accountSid: string;
        status: AnonymizeStatus;
        identity: string;
        dateCreated: Date;
        dateUpdated: Date;
        startTime: Date;
        endTime: Date;
        duration: number;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AnonymizeSolution {
    roomSid: string;
    sid: string;
}
export interface AnonymizeListInstance {
    _version: V1;
    _solution: AnonymizeSolution;
    _uri: string;
    (): AnonymizeContext;
    get(): AnonymizeContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AnonymizeListInstance(version: V1, roomSid: string, sid: string): AnonymizeListInstance;
export {};
