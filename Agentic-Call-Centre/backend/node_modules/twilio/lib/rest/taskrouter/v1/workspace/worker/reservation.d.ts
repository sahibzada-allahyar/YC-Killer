/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
export type ReservationCallStatus = "initiated" | "ringing" | "answered" | "completed";
export type ReservationConferenceEvent = "start" | "end" | "join" | "leave" | "mute" | "hold" | "speaker";
export type ReservationStatus = "pending" | "accepted" | "rejected" | "timeout" | "canceled" | "rescinded" | "wrapping" | "completed";
/**
 * Options to pass to update a ReservationInstance
 */
export interface ReservationContextUpdateOptions {
    /** The If-Match HTTP request header */
    ifMatch?: string;
    /**  */
    reservationStatus?: ReservationStatus;
    /** The new worker activity SID if rejecting a reservation. */
    workerActivitySid?: string;
    /** The assignment instruction for the reservation. */
    instruction?: string;
    /** The SID of the Activity resource to start after executing a Dequeue instruction. */
    dequeuePostWorkActivitySid?: string;
    /** The caller ID of the call to the worker when executing a Dequeue instruction. */
    dequeueFrom?: string;
    /** Whether to record both legs of a call when executing a Dequeue instruction or which leg to record. */
    dequeueRecord?: string;
    /** The timeout for call when executing a Dequeue instruction. */
    dequeueTimeout?: number;
    /** The contact URI of the worker when executing a Dequeue instruction. Can be the URI of the Twilio Client, the SIP URI for Programmable SIP, or the [E.164](https://www.twilio.com/docs/glossary/what-e164) formatted phone number, depending on the destination. */
    dequeueTo?: string;
    /** The callback URL for completed call event when executing a Dequeue instruction. */
    dequeueStatusCallbackUrl?: string;
    /** The Caller ID of the outbound call when executing a Call instruction. */
    callFrom?: string;
    /** Whether to record both legs of a call when executing a Call instruction. */
    callRecord?: string;
    /** The timeout for a call when executing a Call instruction. */
    callTimeout?: number;
    /** The contact URI of the worker when executing a Call instruction. Can be the URI of the Twilio Client, the SIP URI for Programmable SIP, or the [E.164](https://www.twilio.com/docs/glossary/what-e164) formatted phone number, depending on the destination. */
    callTo?: string;
    /** TwiML URI executed on answering the worker\\\'s leg as a result of the Call instruction. */
    callUrl?: string;
    /** The URL to call for the completed call event when executing a Call instruction. */
    callStatusCallbackUrl?: string;
    /** Whether to accept a reservation when executing a Call instruction. */
    callAccept?: boolean;
    /** The Call SID of the call parked in the queue when executing a Redirect instruction. */
    redirectCallSid?: string;
    /** Whether the reservation should be accepted when executing a Redirect instruction. */
    redirectAccept?: boolean;
    /** TwiML URI to redirect the call to when executing the Redirect instruction. */
    redirectUrl?: string;
    /** The Contact URI of the worker when executing a Conference instruction. Can be the URI of the Twilio Client, the SIP URI for Programmable SIP, or the [E.164](https://www.twilio.com/docs/glossary/what-e164) formatted phone number, depending on the destination. */
    to?: string;
    /** The caller ID of the call to the worker when executing a Conference instruction. */
    from?: string;
    /** The URL we should call using the `status_callback_method` to send status information to your application. */
    statusCallback?: string;
    /** The HTTP method we should use to call `status_callback`. Can be: `POST` or `GET` and the default is `POST`. */
    statusCallbackMethod?: string;
    /** The call progress events that we will send to `status_callback`. Can be: `initiated`, `ringing`, `answered`, or `completed`. */
    statusCallbackEvent?: Array<ReservationCallStatus>;
    /** The timeout for a call when executing a Conference instruction. */
    timeout?: number;
    /** Whether to record the participant and their conferences, including the time between conferences. Can be `true` or `false` and the default is `false`. */
    record?: boolean;
    /** Whether the agent is muted in the conference. Defaults to `false`. */
    muted?: boolean;
    /** Whether to play a notification beep when the participant joins or when to play a beep. Can be: `true`, `false`, `onEnter`, or `onExit`. The default value is `true`. */
    beep?: string;
    /** Whether to start the conference when the participant joins, if it has not already started. Can be: `true` or `false` and the default is `true`. If `false` and the conference has not started, the participant is muted and hears background music until another participant starts the conference. */
    startConferenceOnEnter?: boolean;
    /** Whether to end the conference when the agent leaves. */
    endConferenceOnExit?: boolean;
    /** The URL we should call using the `wait_method` for the music to play while participants are waiting for the conference to start. The default value is the URL of our standard hold music. [Learn more about hold music](https://www.twilio.com/labs/twimlets/holdmusic). */
    waitUrl?: string;
    /** The HTTP method we should use to call `wait_url`. Can be `GET` or `POST` and the default is `POST`. When using a static audio file, this should be `GET` so that we can cache the file. */
    waitMethod?: string;
    /** Whether to allow an agent to hear the state of the outbound call, including ringing or disconnect messages. The default is `true`. */
    earlyMedia?: boolean;
    /** The maximum number of participants allowed in the conference. Can be a positive integer from `2` to `250`. The default value is `250`. */
    maxParticipants?: number;
    /** The URL we should call using the `conference_status_callback_method` when the conference events in `conference_status_callback_event` occur. Only the value set by the first participant to join the conference is used. Subsequent `conference_status_callback` values are ignored. */
    conferenceStatusCallback?: string;
    /** The HTTP method we should use to call `conference_status_callback`. Can be: `GET` or `POST` and defaults to `POST`. */
    conferenceStatusCallbackMethod?: string;
    /** The conference status events that we will send to `conference_status_callback`. Can be: `start`, `end`, `join`, `leave`, `mute`, `hold`, `speaker`. */
    conferenceStatusCallbackEvent?: Array<ReservationConferenceEvent>;
    /** Whether to record the conference the participant is joining or when to record the conference. Can be: `true`, `false`, `record-from-start`, and `do-not-record`. The default value is `false`. */
    conferenceRecord?: string;
    /** Whether to trim leading and trailing silence from your recorded conference audio files. Can be: `trim-silence` or `do-not-trim` and defaults to `trim-silence`. */
    conferenceTrim?: string;
    /** The recording channels for the final recording. Can be: `mono` or `dual` and the default is `mono`. */
    recordingChannels?: string;
    /** The URL that we should call using the `recording_status_callback_method` when the recording status changes. */
    recordingStatusCallback?: string;
    /** The HTTP method we should use when we call `recording_status_callback`. Can be: `GET` or `POST` and defaults to `POST`. */
    recordingStatusCallbackMethod?: string;
    /** The URL we should call using the `conference_recording_status_callback_method` when the conference recording is available. */
    conferenceRecordingStatusCallback?: string;
    /** The HTTP method we should use to call `conference_recording_status_callback`. Can be: `GET` or `POST` and defaults to `POST`. */
    conferenceRecordingStatusCallbackMethod?: string;
    /** The [region](https://support.twilio.com/hc/en-us/articles/223132167-How-global-low-latency-routing-and-region-selection-work-for-conferences-and-Client-calls) where we should mix the recorded audio. Can be:`us1`, `ie1`, `de1`, `sg1`, `br1`, `au1`, or `jp1`. */
    region?: string;
    /** The SIP username used for authentication. */
    sipAuthUsername?: string;
    /** The SIP password for authentication. */
    sipAuthPassword?: string;
    /** The call progress events sent via webhooks as a result of a Dequeue instruction. */
    dequeueStatusCallbackEvent?: Array<string>;
    /** The new worker activity SID after executing a Conference instruction. */
    postWorkActivitySid?: string;
    /** Whether to end the conference when the customer leaves. */
    endConferenceOnCustomerExit?: boolean;
    /** Whether to play a notification beep when the customer joins. */
    beepOnCustomerEntrance?: boolean;
    /** The jitter buffer size for conference. Can be: `small`, `medium`, `large`, `off`. */
    jitterBufferSize?: string;
}
/**
 * Options to pass to each
 */
export interface ReservationListInstanceEachOptions {
    /** Returns the list of reservations for a worker with a specified ReservationStatus. Can be: `pending`, `accepted`, `rejected`, `timeout`, `canceled`, or `rescinded`. */
    reservationStatus?: ReservationStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ReservationInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ReservationListInstanceOptions {
    /** Returns the list of reservations for a worker with a specified ReservationStatus. Can be: `pending`, `accepted`, `rejected`, `timeout`, `canceled`, or `rescinded`. */
    reservationStatus?: ReservationStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ReservationListInstancePageOptions {
    /** Returns the list of reservations for a worker with a specified ReservationStatus. Can be: `pending`, `accepted`, `rejected`, `timeout`, `canceled`, or `rescinded`. */
    reservationStatus?: ReservationStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ReservationContext {
    /**
     * Fetch a ReservationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ReservationInstance
     */
    fetch(callback?: (error: Error | null, item?: ReservationInstance) => any): Promise<ReservationInstance>;
    /**
     * Update a ReservationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ReservationInstance
     */
    update(callback?: (error: Error | null, item?: ReservationInstance) => any): Promise<ReservationInstance>;
    /**
     * Update a ReservationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ReservationInstance
     */
    update(params: ReservationContextUpdateOptions, callback?: (error: Error | null, item?: ReservationInstance) => any): Promise<ReservationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ReservationContextSolution {
    workspaceSid: string;
    workerSid: string;
    sid: string;
}
export declare class ReservationContextImpl implements ReservationContext {
    protected _version: V1;
    protected _solution: ReservationContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string, workerSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: ReservationInstance) => any): Promise<ReservationInstance>;
    update(params?: ReservationContextUpdateOptions | ((error: Error | null, item?: ReservationInstance) => any), callback?: (error: Error | null, item?: ReservationInstance) => any): Promise<ReservationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ReservationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ReservationPayload extends TwilioResponsePayload {
    reservations: ReservationResource[];
}
interface ReservationResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    reservation_status: ReservationStatus;
    sid: string;
    task_sid: string;
    worker_name: string;
    worker_sid: string;
    workspace_sid: string;
    url: string;
    links: Record<string, string>;
}
export declare class ReservationInstance {
    protected _version: V1;
    protected _solution: ReservationContextSolution;
    protected _context?: ReservationContext;
    constructor(_version: V1, payload: ReservationResource, workspaceSid: string, workerSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the WorkerReservation resource.
     */
    accountSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    reservationStatus: ReservationStatus;
    /**
     * The unique string that we created to identify the WorkerReservation resource.
     */
    sid: string;
    /**
     * The SID of the reserved Task resource.
     */
    taskSid: string;
    /**
     * The `friendly_name` of the Worker that is reserved.
     */
    workerName: string;
    /**
     * The SID of the reserved Worker resource.
     */
    workerSid: string;
    /**
     * The SID of the Workspace that this worker is contained within.
     */
    workspaceSid: string;
    /**
     * The absolute URL of the WorkerReservation resource.
     */
    url: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a ReservationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ReservationInstance
     */
    fetch(callback?: (error: Error | null, item?: ReservationInstance) => any): Promise<ReservationInstance>;
    /**
     * Update a ReservationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ReservationInstance
     */
    update(callback?: (error: Error | null, item?: ReservationInstance) => any): Promise<ReservationInstance>;
    /**
     * Update a ReservationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ReservationInstance
     */
    update(params: ReservationContextUpdateOptions, callback?: (error: Error | null, item?: ReservationInstance) => any): Promise<ReservationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        reservationStatus: ReservationStatus;
        sid: string;
        taskSid: string;
        workerName: string;
        workerSid: string;
        workspaceSid: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ReservationSolution {
    workspaceSid: string;
    workerSid: string;
}
export interface ReservationListInstance {
    _version: V1;
    _solution: ReservationSolution;
    _uri: string;
    (sid: string): ReservationContext;
    get(sid: string): ReservationContext;
    /**
     * Streams ReservationInstance records from the API.
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
     * @param { ReservationListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ReservationInstance, done: (err?: Error) => void) => void): void;
    each(params: ReservationListInstanceEachOptions, callback?: (item: ReservationInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ReservationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ReservationPage) => any): Promise<ReservationPage>;
    /**
     * Lists ReservationInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ReservationListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ReservationInstance[]) => any): Promise<ReservationInstance[]>;
    list(params: ReservationListInstanceOptions, callback?: (error: Error | null, items: ReservationInstance[]) => any): Promise<ReservationInstance[]>;
    /**
     * Retrieve a single page of ReservationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ReservationListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ReservationPage) => any): Promise<ReservationPage>;
    page(params: ReservationListInstancePageOptions, callback?: (error: Error | null, items: ReservationPage) => any): Promise<ReservationPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ReservationListInstance(version: V1, workspaceSid: string, workerSid: string): ReservationListInstance;
export declare class ReservationPage extends Page<V1, ReservationPayload, ReservationResource, ReservationInstance> {
    /**
     * Initialize the ReservationPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ReservationSolution);
    /**
     * Build an instance of ReservationInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ReservationResource): ReservationInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
