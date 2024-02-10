/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { ConferenceParticipantListInstance } from "./conference/conferenceParticipant";
export type ConferenceConferenceEndReason = "last_participant_left" | "conference_ended_via_api" | "participant_with_end_conference_on_exit_left" | "last_participant_kicked" | "participant_with_end_conference_on_exit_kicked";
export type ConferenceConferenceStatus = "in_progress" | "not_started" | "completed" | "summary_timeout";
export type ConferenceProcessingState = "complete" | "in_progress" | "timeout";
export type ConferenceRegion = "us1" | "au1" | "br1" | "ie1" | "jp1" | "sg1" | "de1";
export type ConferenceTag = "invalid_requested_region" | "duplicate_identity" | "start_failure" | "region_configuration_issues" | "quality_warnings" | "participant_behavior_issues" | "high_packet_loss" | "high_jitter" | "high_latency" | "low_mos" | "detected_silence";
/**
 * Options to pass to each
 */
export interface ConferenceListInstanceEachOptions {
    /** The SID of the conference. */
    conferenceSid?: string;
    /** Custom label for the conference resource, up to 64 characters. */
    friendlyName?: string;
    /** Conference status. */
    status?: string;
    /** Conferences created after the provided timestamp specified in ISO 8601 format */
    createdAfter?: string;
    /** Conferences created before the provided timestamp specified in ISO 8601 format. */
    createdBefore?: string;
    /** Twilio region where the conference media was mixed. */
    mixerRegion?: string;
    /** Tags applied by Twilio for common potential configuration, quality, or performance issues. */
    tags?: string;
    /** Account SID for the subaccount whose resources you wish to retrieve. */
    subaccount?: string;
    /** Potential configuration, behavior, or performance issues detected during the conference. */
    detectedIssues?: string;
    /** Conference end reason; e.g. last participant left, modified by API, etc. */
    endReason?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ConferenceInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ConferenceListInstanceOptions {
    /** The SID of the conference. */
    conferenceSid?: string;
    /** Custom label for the conference resource, up to 64 characters. */
    friendlyName?: string;
    /** Conference status. */
    status?: string;
    /** Conferences created after the provided timestamp specified in ISO 8601 format */
    createdAfter?: string;
    /** Conferences created before the provided timestamp specified in ISO 8601 format. */
    createdBefore?: string;
    /** Twilio region where the conference media was mixed. */
    mixerRegion?: string;
    /** Tags applied by Twilio for common potential configuration, quality, or performance issues. */
    tags?: string;
    /** Account SID for the subaccount whose resources you wish to retrieve. */
    subaccount?: string;
    /** Potential configuration, behavior, or performance issues detected during the conference. */
    detectedIssues?: string;
    /** Conference end reason; e.g. last participant left, modified by API, etc. */
    endReason?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ConferenceListInstancePageOptions {
    /** The SID of the conference. */
    conferenceSid?: string;
    /** Custom label for the conference resource, up to 64 characters. */
    friendlyName?: string;
    /** Conference status. */
    status?: string;
    /** Conferences created after the provided timestamp specified in ISO 8601 format */
    createdAfter?: string;
    /** Conferences created before the provided timestamp specified in ISO 8601 format. */
    createdBefore?: string;
    /** Twilio region where the conference media was mixed. */
    mixerRegion?: string;
    /** Tags applied by Twilio for common potential configuration, quality, or performance issues. */
    tags?: string;
    /** Account SID for the subaccount whose resources you wish to retrieve. */
    subaccount?: string;
    /** Potential configuration, behavior, or performance issues detected during the conference. */
    detectedIssues?: string;
    /** Conference end reason; e.g. last participant left, modified by API, etc. */
    endReason?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ConferenceContext {
    conferenceParticipants: ConferenceParticipantListInstance;
    /**
     * Fetch a ConferenceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConferenceInstance
     */
    fetch(callback?: (error: Error | null, item?: ConferenceInstance) => any): Promise<ConferenceInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ConferenceContextSolution {
    conferenceSid: string;
}
export declare class ConferenceContextImpl implements ConferenceContext {
    protected _version: V1;
    protected _solution: ConferenceContextSolution;
    protected _uri: string;
    protected _conferenceParticipants?: ConferenceParticipantListInstance;
    constructor(_version: V1, conferenceSid: string);
    get conferenceParticipants(): ConferenceParticipantListInstance;
    fetch(callback?: (error: Error | null, item?: ConferenceInstance) => any): Promise<ConferenceInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ConferenceContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ConferencePayload extends TwilioResponsePayload {
    conferences: ConferenceResource[];
}
interface ConferenceResource {
    conference_sid: string;
    account_sid: string;
    friendly_name: string;
    create_time: Date;
    start_time: Date;
    end_time: Date;
    duration_seconds: number;
    connect_duration_seconds: number;
    status: ConferenceConferenceStatus;
    max_participants: number;
    max_concurrent_participants: number;
    unique_participants: number;
    end_reason: ConferenceConferenceEndReason;
    ended_by: string;
    mixer_region: ConferenceRegion;
    mixer_region_requested: ConferenceRegion;
    recording_enabled: boolean;
    detected_issues: any;
    tags: Array<ConferenceTag>;
    tag_info: any;
    processing_state: ConferenceProcessingState;
    url: string;
    links: Record<string, string>;
}
export declare class ConferenceInstance {
    protected _version: V1;
    protected _solution: ConferenceContextSolution;
    protected _context?: ConferenceContext;
    constructor(_version: V1, payload: ConferenceResource, conferenceSid?: string);
    /**
     * The unique SID identifier of the Conference.
     */
    conferenceSid: string;
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * Custom label for the conference resource, up to 64 characters.
     */
    friendlyName: string;
    /**
     * Conference creation date and time in ISO 8601 format.
     */
    createTime: Date;
    /**
     * Timestamp in ISO 8601 format when the conference started. Conferences do not start until at least two participants join, at least one of whom has startConferenceOnEnter=true.
     */
    startTime: Date;
    /**
     * Conference end date and time in ISO 8601 format.
     */
    endTime: Date;
    /**
     * Conference duration in seconds.
     */
    durationSeconds: number;
    /**
     * Duration of the between conference start event and conference end event in seconds.
     */
    connectDurationSeconds: number;
    status: ConferenceConferenceStatus;
    /**
     * Maximum number of concurrent participants as specified by the configuration.
     */
    maxParticipants: number;
    /**
     * Actual maximum number of concurrent participants in the conference.
     */
    maxConcurrentParticipants: number;
    /**
     * Unique conference participants based on caller ID.
     */
    uniqueParticipants: number;
    endReason: ConferenceConferenceEndReason;
    /**
     * Call SID of the participant whose actions ended the conference.
     */
    endedBy: string;
    mixerRegion: ConferenceRegion;
    mixerRegionRequested: ConferenceRegion;
    /**
     * Boolean. Indicates whether recording was enabled at the conference mixer.
     */
    recordingEnabled: boolean;
    /**
     * Potential issues detected by Twilio during the conference.
     */
    detectedIssues: any;
    /**
     * Tags for detected conference conditions and participant behaviors which may be of interest.
     */
    tags: Array<ConferenceTag>;
    /**
     * Object. Contains details about conference tags including severity.
     */
    tagInfo: any;
    processingState: ConferenceProcessingState;
    /**
     * The URL of this resource.
     */
    url: string;
    /**
     * Contains a dictionary of URL links to nested resources of this Conference.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a ConferenceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConferenceInstance
     */
    fetch(callback?: (error: Error | null, item?: ConferenceInstance) => any): Promise<ConferenceInstance>;
    /**
     * Access the conferenceParticipants.
     */
    conferenceParticipants(): ConferenceParticipantListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        conferenceSid: string;
        accountSid: string;
        friendlyName: string;
        createTime: Date;
        startTime: Date;
        endTime: Date;
        durationSeconds: number;
        connectDurationSeconds: number;
        status: ConferenceConferenceStatus;
        maxParticipants: number;
        maxConcurrentParticipants: number;
        uniqueParticipants: number;
        endReason: ConferenceConferenceEndReason;
        endedBy: string;
        mixerRegion: ConferenceRegion;
        mixerRegionRequested: ConferenceRegion;
        recordingEnabled: boolean;
        detectedIssues: any;
        tags: ConferenceTag[];
        tagInfo: any;
        processingState: ConferenceProcessingState;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ConferenceSolution {
}
export interface ConferenceListInstance {
    _version: V1;
    _solution: ConferenceSolution;
    _uri: string;
    (conferenceSid: string): ConferenceContext;
    get(conferenceSid: string): ConferenceContext;
    /**
     * Streams ConferenceInstance records from the API.
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
     * @param { ConferenceListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ConferenceInstance, done: (err?: Error) => void) => void): void;
    each(params: ConferenceListInstanceEachOptions, callback?: (item: ConferenceInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ConferenceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ConferencePage) => any): Promise<ConferencePage>;
    /**
     * Lists ConferenceInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ConferenceListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ConferenceInstance[]) => any): Promise<ConferenceInstance[]>;
    list(params: ConferenceListInstanceOptions, callback?: (error: Error | null, items: ConferenceInstance[]) => any): Promise<ConferenceInstance[]>;
    /**
     * Retrieve a single page of ConferenceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ConferenceListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ConferencePage) => any): Promise<ConferencePage>;
    page(params: ConferenceListInstancePageOptions, callback?: (error: Error | null, items: ConferencePage) => any): Promise<ConferencePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ConferenceListInstance(version: V1): ConferenceListInstance;
export declare class ConferencePage extends Page<V1, ConferencePayload, ConferenceResource, ConferenceInstance> {
    /**
     * Initialize the ConferencePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ConferenceSolution);
    /**
     * Build an instance of ConferenceInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ConferenceResource): ConferenceInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
