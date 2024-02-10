/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { AnonymizeListInstance } from "./participant/anonymize";
import { PublishedTrackListInstance } from "./participant/publishedTrack";
import { SubscribeRulesListInstance } from "./participant/subscribeRules";
import { SubscribedTrackListInstance } from "./participant/subscribedTrack";
export type ParticipantStatus = "connected" | "disconnected";
/**
 * Options to pass to update a ParticipantInstance
 */
export interface ParticipantContextUpdateOptions {
    /**  */
    status?: ParticipantStatus;
}
/**
 * Options to pass to each
 */
export interface ParticipantListInstanceEachOptions {
    /** Read only the participants with this status. Can be: `connected` or `disconnected`. For `in-progress` Rooms the default Status is `connected`, for `completed` Rooms only `disconnected` Participants are returned. */
    status?: ParticipantStatus;
    /** Read only the Participants with this [User](https://www.twilio.com/docs/chat/rest/user-resource) `identity` value. */
    identity?: string;
    /** Read only Participants that started after this date in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#UTC) format. */
    dateCreatedAfter?: Date;
    /** Read only Participants that started before this date in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#UTC) format. */
    dateCreatedBefore?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ParticipantInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ParticipantListInstanceOptions {
    /** Read only the participants with this status. Can be: `connected` or `disconnected`. For `in-progress` Rooms the default Status is `connected`, for `completed` Rooms only `disconnected` Participants are returned. */
    status?: ParticipantStatus;
    /** Read only the Participants with this [User](https://www.twilio.com/docs/chat/rest/user-resource) `identity` value. */
    identity?: string;
    /** Read only Participants that started after this date in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#UTC) format. */
    dateCreatedAfter?: Date;
    /** Read only Participants that started before this date in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#UTC) format. */
    dateCreatedBefore?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ParticipantListInstancePageOptions {
    /** Read only the participants with this status. Can be: `connected` or `disconnected`. For `in-progress` Rooms the default Status is `connected`, for `completed` Rooms only `disconnected` Participants are returned. */
    status?: ParticipantStatus;
    /** Read only the Participants with this [User](https://www.twilio.com/docs/chat/rest/user-resource) `identity` value. */
    identity?: string;
    /** Read only Participants that started after this date in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#UTC) format. */
    dateCreatedAfter?: Date;
    /** Read only Participants that started before this date in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601#UTC) format. */
    dateCreatedBefore?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ParticipantContext {
    anonymize: AnonymizeListInstance;
    publishedTracks: PublishedTrackListInstance;
    subscribeRules: SubscribeRulesListInstance;
    subscribedTracks: SubscribedTrackListInstance;
    /**
     * Fetch a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    fetch(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Update a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    update(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Update a ParticipantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    update(params: ParticipantContextUpdateOptions, callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ParticipantContextSolution {
    roomSid: string;
    sid: string;
}
export declare class ParticipantContextImpl implements ParticipantContext {
    protected _version: V1;
    protected _solution: ParticipantContextSolution;
    protected _uri: string;
    protected _anonymize?: AnonymizeListInstance;
    protected _publishedTracks?: PublishedTrackListInstance;
    protected _subscribeRules?: SubscribeRulesListInstance;
    protected _subscribedTracks?: SubscribedTrackListInstance;
    constructor(_version: V1, roomSid: string, sid: string);
    get anonymize(): AnonymizeListInstance;
    get publishedTracks(): PublishedTrackListInstance;
    get subscribeRules(): SubscribeRulesListInstance;
    get subscribedTracks(): SubscribedTrackListInstance;
    fetch(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    update(params?: ParticipantContextUpdateOptions | ((error: Error | null, item?: ParticipantInstance) => any), callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ParticipantContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ParticipantPayload extends TwilioResponsePayload {
    participants: ParticipantResource[];
}
interface ParticipantResource {
    sid: string;
    room_sid: string;
    account_sid: string;
    status: ParticipantStatus;
    identity: string;
    date_created: Date;
    date_updated: Date;
    start_time: Date;
    end_time: Date;
    duration: number;
    url: string;
    links: Record<string, string>;
}
export declare class ParticipantInstance {
    protected _version: V1;
    protected _solution: ParticipantContextSolution;
    protected _context?: ParticipantContext;
    constructor(_version: V1, payload: ParticipantResource, roomSid: string, sid?: string);
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
    status: ParticipantStatus;
    /**
     * The application-defined string that uniquely identifies the resource\'s User within a Room. If a client joins with an existing Identity, the existing client is disconnected. See [access tokens](https://www.twilio.com/docs/video/tutorials/user-identity-access-tokens) and [limits](https://www.twilio.com/docs/video/programmable-video-limits) for more info.
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
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    fetch(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Update a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    update(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Update a ParticipantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    update(params: ParticipantContextUpdateOptions, callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Access the anonymize.
     */
    anonymize(): AnonymizeListInstance;
    /**
     * Access the publishedTracks.
     */
    publishedTracks(): PublishedTrackListInstance;
    /**
     * Access the subscribeRules.
     */
    subscribeRules(): SubscribeRulesListInstance;
    /**
     * Access the subscribedTracks.
     */
    subscribedTracks(): SubscribedTrackListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        roomSid: string;
        accountSid: string;
        status: ParticipantStatus;
        identity: string;
        dateCreated: Date;
        dateUpdated: Date;
        startTime: Date;
        endTime: Date;
        duration: number;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ParticipantSolution {
    roomSid: string;
}
export interface ParticipantListInstance {
    _version: V1;
    _solution: ParticipantSolution;
    _uri: string;
    (sid: string): ParticipantContext;
    get(sid: string): ParticipantContext;
    /**
     * Streams ParticipantInstance records from the API.
     *
     * This operation lazily loads records as efficiently as possible until the limit
     * is reached.
     *
     * The results are passed into the callback function, so this operation is memory
     * efficient.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ParticipantListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ParticipantInstance, done: (err?: Error) => void) => void): void;
    each(params: ParticipantListInstanceEachOptions, callback?: (item: ParticipantInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ParticipantInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ParticipantPage) => any): Promise<ParticipantPage>;
    /**
     * Lists ParticipantInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ParticipantListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ParticipantInstance[]) => any): Promise<ParticipantInstance[]>;
    list(params: ParticipantListInstanceOptions, callback?: (error: Error | null, items: ParticipantInstance[]) => any): Promise<ParticipantInstance[]>;
    /**
     * Retrieve a single page of ParticipantInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ParticipantListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ParticipantPage) => any): Promise<ParticipantPage>;
    page(params: ParticipantListInstancePageOptions, callback?: (error: Error | null, items: ParticipantPage) => any): Promise<ParticipantPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ParticipantListInstance(version: V1, roomSid: string): ParticipantListInstance;
export declare class ParticipantPage extends Page<V1, ParticipantPayload, ParticipantResource, ParticipantInstance> {
    /**
     * Initialize the ParticipantPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ParticipantSolution);
    /**
     * Build an instance of ParticipantInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ParticipantResource): ParticipantInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
