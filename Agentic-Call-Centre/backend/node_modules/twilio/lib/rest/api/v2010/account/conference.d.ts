/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
import { ParticipantListInstance } from "./conference/participant";
import { RecordingListInstance } from "./conference/recording";
export type ConferenceReasonConferenceEnded = "conference-ended-via-api" | "participant-with-end-conference-on-exit-left" | "participant-with-end-conference-on-exit-kicked" | "last-participant-kicked" | "last-participant-left";
export type ConferenceStatus = "init" | "in-progress" | "completed";
export type ConferenceUpdateStatus = "completed";
/**
 * Options to pass to update a ConferenceInstance
 */
export interface ConferenceContextUpdateOptions {
    /**  */
    status?: ConferenceUpdateStatus;
    /** The URL we should call to announce something into the conference. The URL may return an MP3 file, a WAV file, or a TwiML document that contains `<Play>`, `<Say>`, `<Pause>`, or `<Redirect>` verbs. */
    announceUrl?: string;
    /** The HTTP method used to call `announce_url`. Can be: `GET` or `POST` and the default is `POST` */
    announceMethod?: string;
}
/**
 * Options to pass to each
 */
export interface ConferenceListInstanceEachOptions {
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that started on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify  conferences that started on or after midnight on a date, use `>=YYYY-MM-DD`. */
    dateCreated?: Date;
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that started on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify  conferences that started on or after midnight on a date, use `>=YYYY-MM-DD`. */
    dateCreatedBefore?: Date;
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that started on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify  conferences that started on or after midnight on a date, use `>=YYYY-MM-DD`. */
    dateCreatedAfter?: Date;
    /** The `date_updated` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that were last updated on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify conferences that were last updated on or after midnight on a given date, use  `>=YYYY-MM-DD`. */
    dateUpdated?: Date;
    /** The `date_updated` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that were last updated on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify conferences that were last updated on or after midnight on a given date, use  `>=YYYY-MM-DD`. */
    dateUpdatedBefore?: Date;
    /** The `date_updated` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that were last updated on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify conferences that were last updated on or after midnight on a given date, use  `>=YYYY-MM-DD`. */
    dateUpdatedAfter?: Date;
    /** The string that identifies the Conference resources to read. */
    friendlyName?: string;
    /** The status of the resources to read. Can be: `init`, `in-progress`, or `completed`. */
    status?: ConferenceStatus;
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
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that started on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify  conferences that started on or after midnight on a date, use `>=YYYY-MM-DD`. */
    dateCreated?: Date;
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that started on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify  conferences that started on or after midnight on a date, use `>=YYYY-MM-DD`. */
    dateCreatedBefore?: Date;
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that started on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify  conferences that started on or after midnight on a date, use `>=YYYY-MM-DD`. */
    dateCreatedAfter?: Date;
    /** The `date_updated` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that were last updated on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify conferences that were last updated on or after midnight on a given date, use  `>=YYYY-MM-DD`. */
    dateUpdated?: Date;
    /** The `date_updated` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that were last updated on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify conferences that were last updated on or after midnight on a given date, use  `>=YYYY-MM-DD`. */
    dateUpdatedBefore?: Date;
    /** The `date_updated` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that were last updated on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify conferences that were last updated on or after midnight on a given date, use  `>=YYYY-MM-DD`. */
    dateUpdatedAfter?: Date;
    /** The string that identifies the Conference resources to read. */
    friendlyName?: string;
    /** The status of the resources to read. Can be: `init`, `in-progress`, or `completed`. */
    status?: ConferenceStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ConferenceListInstancePageOptions {
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that started on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify  conferences that started on or after midnight on a date, use `>=YYYY-MM-DD`. */
    dateCreated?: Date;
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that started on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify  conferences that started on or after midnight on a date, use `>=YYYY-MM-DD`. */
    dateCreatedBefore?: Date;
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that started on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify  conferences that started on or after midnight on a date, use `>=YYYY-MM-DD`. */
    dateCreatedAfter?: Date;
    /** The `date_updated` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that were last updated on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify conferences that were last updated on or after midnight on a given date, use  `>=YYYY-MM-DD`. */
    dateUpdated?: Date;
    /** The `date_updated` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that were last updated on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify conferences that were last updated on or after midnight on a given date, use  `>=YYYY-MM-DD`. */
    dateUpdatedBefore?: Date;
    /** The `date_updated` value, specified as `YYYY-MM-DD`, of the resources to read. To read conferences that were last updated on or before midnight on a date, use `<=YYYY-MM-DD`, and to specify conferences that were last updated on or after midnight on a given date, use  `>=YYYY-MM-DD`. */
    dateUpdatedAfter?: Date;
    /** The string that identifies the Conference resources to read. */
    friendlyName?: string;
    /** The status of the resources to read. Can be: `init`, `in-progress`, or `completed`. */
    status?: ConferenceStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ConferenceContext {
    participants: ParticipantListInstance;
    recordings: RecordingListInstance;
    /**
     * Fetch a ConferenceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConferenceInstance
     */
    fetch(callback?: (error: Error | null, item?: ConferenceInstance) => any): Promise<ConferenceInstance>;
    /**
     * Update a ConferenceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConferenceInstance
     */
    update(callback?: (error: Error | null, item?: ConferenceInstance) => any): Promise<ConferenceInstance>;
    /**
     * Update a ConferenceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConferenceInstance
     */
    update(params: ConferenceContextUpdateOptions, callback?: (error: Error | null, item?: ConferenceInstance) => any): Promise<ConferenceInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ConferenceContextSolution {
    accountSid: string;
    sid: string;
}
export declare class ConferenceContextImpl implements ConferenceContext {
    protected _version: V2010;
    protected _solution: ConferenceContextSolution;
    protected _uri: string;
    protected _participants?: ParticipantListInstance;
    protected _recordings?: RecordingListInstance;
    constructor(_version: V2010, accountSid: string, sid: string);
    get participants(): ParticipantListInstance;
    get recordings(): RecordingListInstance;
    fetch(callback?: (error: Error | null, item?: ConferenceInstance) => any): Promise<ConferenceInstance>;
    update(params?: ConferenceContextUpdateOptions | ((error: Error | null, item?: ConferenceInstance) => any), callback?: (error: Error | null, item?: ConferenceInstance) => any): Promise<ConferenceInstance>;
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
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    api_version: string;
    friendly_name: string;
    region: string;
    sid: string;
    status: ConferenceStatus;
    uri: string;
    subresource_uris: Record<string, string>;
    reason_conference_ended: ConferenceReasonConferenceEnded;
    call_sid_ending_conference: string;
}
export declare class ConferenceInstance {
    protected _version: V2010;
    protected _solution: ConferenceContextSolution;
    protected _context?: ConferenceContext;
    constructor(_version: V2010, payload: ConferenceResource, accountSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created this Conference resource.
     */
    accountSid: string;
    /**
     * The date and time in GMT that this resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that this resource was last updated, specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The API version used to create this conference.
     */
    apiVersion: string;
    /**
     * A string that you assigned to describe this conference room. Maxiumum length is 128 characters.
     */
    friendlyName: string;
    /**
     * A string that represents the Twilio Region where the conference audio was mixed. May be `us1`, `ie1`,  `de1`, `sg1`, `br1`, `au1`, and `jp1`. Basic conference audio will always be mixed in `us1`. Global Conference audio will be mixed nearest to the majority of participants.
     */
    region: string;
    /**
     * The unique string that that we created to identify this Conference resource.
     */
    sid: string;
    status: ConferenceStatus;
    /**
     * The URI of this resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    /**
     * A list of related resources identified by their URIs relative to `https://api.twilio.com`.
     */
    subresourceUris: Record<string, string>;
    reasonConferenceEnded: ConferenceReasonConferenceEnded;
    /**
     * The call SID that caused the conference to end.
     */
    callSidEndingConference: string;
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
     * Update a ConferenceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConferenceInstance
     */
    update(callback?: (error: Error | null, item?: ConferenceInstance) => any): Promise<ConferenceInstance>;
    /**
     * Update a ConferenceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConferenceInstance
     */
    update(params: ConferenceContextUpdateOptions, callback?: (error: Error | null, item?: ConferenceInstance) => any): Promise<ConferenceInstance>;
    /**
     * Access the participants.
     */
    participants(): ParticipantListInstance;
    /**
     * Access the recordings.
     */
    recordings(): RecordingListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        apiVersion: string;
        friendlyName: string;
        region: string;
        sid: string;
        status: ConferenceStatus;
        uri: string;
        subresourceUris: Record<string, string>;
        reasonConferenceEnded: ConferenceReasonConferenceEnded;
        callSidEndingConference: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ConferenceSolution {
    accountSid: string;
}
export interface ConferenceListInstance {
    _version: V2010;
    _solution: ConferenceSolution;
    _uri: string;
    (sid: string): ConferenceContext;
    get(sid: string): ConferenceContext;
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
export declare function ConferenceListInstance(version: V2010, accountSid: string): ConferenceListInstance;
export declare class ConferencePage extends Page<V2010, ConferencePayload, ConferenceResource, ConferenceInstance> {
    /**
     * Initialize the ConferencePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: ConferenceSolution);
    /**
     * Build an instance of ConferenceInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ConferenceResource): ConferenceInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
