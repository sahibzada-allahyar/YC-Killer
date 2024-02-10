/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2010 from "../../../V2010";
export type ParticipantStatus = "queued" | "connecting" | "ringing" | "connected" | "complete" | "failed";
/**
 * Options to pass to update a ParticipantInstance
 */
export interface ParticipantContextUpdateOptions {
    /** Whether the participant should be muted. Can be `true` or `false`. `true` will mute the participant, and `false` will un-mute them. Anything value other than `true` or `false` is interpreted as `false`. */
    muted?: boolean;
    /** Whether the participant should be on hold. Can be: `true` or `false`. `true` puts the participant on hold, and `false` lets them rejoin the conference. */
    hold?: boolean;
    /** The URL we call using the `hold_method` for music that plays when the participant is on hold. The URL may return an MP3 file, a WAV file, or a TwiML document that contains `<Play>`, `<Say>`, `<Pause>`, or `<Redirect>` verbs. */
    holdUrl?: string;
    /** The HTTP method we should use to call `hold_url`. Can be: `GET` or `POST` and the default is `GET`. */
    holdMethod?: string;
    /** The URL we call using the `announce_method` for an announcement to the participant. The URL may return an MP3 file, a WAV file, or a TwiML document that contains `<Play>`, `<Say>`, `<Pause>`, or `<Redirect>` verbs. */
    announceUrl?: string;
    /** The HTTP method we should use to call `announce_url`. Can be: `GET` or `POST` and defaults to `POST`. */
    announceMethod?: string;
    /** The URL we call using the `wait_method` for the music to play while participants are waiting for the conference to start. The URL may return an MP3 file, a WAV file, or a TwiML document that contains `<Play>`, `<Say>`, `<Pause>`, or `<Redirect>` verbs. The default value is the URL of our standard hold music. [Learn more about hold music](https://www.twilio.com/labs/twimlets/holdmusic). */
    waitUrl?: string;
    /** The HTTP method we should use to call `wait_url`. Can be `GET` or `POST` and the default is `POST`. When using a static audio file, this should be `GET` so that we can cache the file. */
    waitMethod?: string;
    /** Whether to play a notification beep to the conference when the participant exits. Can be: `true` or `false`. */
    beepOnExit?: boolean;
    /** Whether to end the conference when the participant leaves. Can be: `true` or `false` and defaults to `false`. */
    endConferenceOnExit?: boolean;
    /** Whether the participant is coaching another call. Can be: `true` or `false`. If not present, defaults to `false` unless `call_sid_to_coach` is defined. If `true`, `call_sid_to_coach` must be defined. */
    coaching?: boolean;
    /** The SID of the participant who is being `coached`. The participant being coached is the only participant who can hear the participant who is `coaching`. */
    callSidToCoach?: string;
}
/**
 * Options to pass to create a ParticipantInstance
 */
export interface ParticipantListInstanceCreateOptions {
    /** The phone number, Client identifier, or username portion of SIP address that made this call. Phone numbers are in [E.164](https://www.twilio.com/docs/glossary/what-e164) format (e.g., +16175551212). Client identifiers are formatted `client:name`. If using a phone number, it must be a Twilio number or a Verified [outgoing caller id](https://www.twilio.com/docs/voice/api/outgoing-caller-ids) for your account. If the `to` parameter is a phone number, `from` must also be a phone number. If `to` is sip address, this value of `from` should be a username portion to be used to populate the P-Asserted-Identity header that is passed to the SIP endpoint. */
    from: string;
    /** The phone number, SIP address, or Client identifier that received this call. Phone numbers are in [E.164](https://www.twilio.com/docs/glossary/what-e164) format (e.g., +16175551212). SIP addresses are formatted as `sip:name@company.com`. Client identifiers are formatted `client:name`. [Custom parameters](https://www.twilio.com/docs/voice/api/conference-participant-resource#custom-parameters) may also be specified. */
    to: string;
    /** The URL we should call using the `status_callback_method` to send status information to your application. */
    statusCallback?: string;
    /** The HTTP method we should use to call `status_callback`. Can be: `GET` and `POST` and defaults to `POST`. */
    statusCallbackMethod?: string;
    /** The conference state changes that should generate a call to `status_callback`. Can be: `initiated`, `ringing`, `answered`, and `completed`. Separate multiple values with a space. The default value is `completed`. */
    statusCallbackEvent?: Array<string>;
    /** A label for this participant. If one is supplied, it may subsequently be used to fetch, update or delete the participant. */
    label?: string;
    /** The number of seconds that we should allow the phone to ring before assuming there is no answer. Can be an integer between `5` and `600`, inclusive. The default value is `60`. We always add a 5-second timeout buffer to outgoing calls, so  value of 10 would result in an actual timeout that was closer to 15 seconds. */
    timeout?: number;
    /** Whether to record the participant and their conferences, including the time between conferences. Can be `true` or `false` and the default is `false`. */
    record?: boolean;
    /** Whether the agent is muted in the conference. Can be `true` or `false` and the default is `false`. */
    muted?: boolean;
    /** Whether to play a notification beep to the conference when the participant joins. Can be: `true`, `false`, `onEnter`, or `onExit`. The default value is `true`. */
    beep?: string;
    /** Whether to start the conference when the participant joins, if it has not already started. Can be: `true` or `false` and the default is `true`. If `false` and the conference has not started, the participant is muted and hears background music until another participant starts the conference. */
    startConferenceOnEnter?: boolean;
    /** Whether to end the conference when the participant leaves. Can be: `true` or `false` and defaults to `false`. */
    endConferenceOnExit?: boolean;
    /** The URL we should call using the `wait_method` for the music to play while participants are waiting for the conference to start. The default value is the URL of our standard hold music. [Learn more about hold music](https://www.twilio.com/labs/twimlets/holdmusic). */
    waitUrl?: string;
    /** The HTTP method we should use to call `wait_url`. Can be `GET` or `POST` and the default is `POST`. When using a static audio file, this should be `GET` so that we can cache the file. */
    waitMethod?: string;
    /** Whether to allow an agent to hear the state of the outbound call, including ringing or disconnect messages. Can be: `true` or `false` and defaults to `true`. */
    earlyMedia?: boolean;
    /** The maximum number of participants in the conference. Can be a positive integer from `2` to `250`. The default value is `250`. */
    maxParticipants?: number;
    /** Whether to record the conference the participant is joining. Can be: `true`, `false`, `record-from-start`, and `do-not-record`. The default value is `false`. */
    conferenceRecord?: string;
    /** Whether to trim leading and trailing silence from the conference recording. Can be: `trim-silence` or `do-not-trim` and defaults to `trim-silence`. */
    conferenceTrim?: string;
    /** The URL we should call using the `conference_status_callback_method` when the conference events in `conference_status_callback_event` occur. Only the value set by the first participant to join the conference is used. Subsequent `conference_status_callback` values are ignored. */
    conferenceStatusCallback?: string;
    /** The HTTP method we should use to call `conference_status_callback`. Can be: `GET` or `POST` and defaults to `POST`. */
    conferenceStatusCallbackMethod?: string;
    /** The conference state changes that should generate a call to `conference_status_callback`. Can be: `start`, `end`, `join`, `leave`, `mute`, `hold`, `modify`, `speaker`, and `announcement`. Separate multiple values with a space. Defaults to `start end`. */
    conferenceStatusCallbackEvent?: Array<string>;
    /** The recording channels for the final recording. Can be: `mono` or `dual` and the default is `mono`. */
    recordingChannels?: string;
    /** The URL that we should call using the `recording_status_callback_method` when the recording status changes. */
    recordingStatusCallback?: string;
    /** The HTTP method we should use when we call `recording_status_callback`. Can be: `GET` or `POST` and defaults to `POST`. */
    recordingStatusCallbackMethod?: string;
    /** The SIP username used for authentication. */
    sipAuthUsername?: string;
    /** The SIP password for authentication. */
    sipAuthPassword?: string;
    /** The [region](https://support.twilio.com/hc/en-us/articles/223132167-How-global-low-latency-routing-and-region-selection-work-for-conferences-and-Client-calls) where we should mix the recorded audio. Can be:`us1`, `ie1`, `de1`, `sg1`, `br1`, `au1`, or `jp1`. */
    region?: string;
    /** The URL we should call using the `conference_recording_status_callback_method` when the conference recording is available. */
    conferenceRecordingStatusCallback?: string;
    /** The HTTP method we should use to call `conference_recording_status_callback`. Can be: `GET` or `POST` and defaults to `POST`. */
    conferenceRecordingStatusCallbackMethod?: string;
    /** The recording state changes that should generate a call to `recording_status_callback`. Can be: `started`, `in-progress`, `paused`, `resumed`, `stopped`, `completed`, `failed`, and `absent`. Separate multiple values with a space, ex: `\\\'in-progress completed failed\\\'`. */
    recordingStatusCallbackEvent?: Array<string>;
    /** The conference recording state changes that generate a call to `conference_recording_status_callback`. Can be: `in-progress`, `completed`, `failed`, and `absent`. Separate multiple values with a space, ex: `\\\'in-progress completed failed\\\'` */
    conferenceRecordingStatusCallbackEvent?: Array<string>;
    /** Whether the participant is coaching another call. Can be: `true` or `false`. If not present, defaults to `false` unless `call_sid_to_coach` is defined. If `true`, `call_sid_to_coach` must be defined. */
    coaching?: boolean;
    /** The SID of the participant who is being `coached`. The participant being coached is the only participant who can hear the participant who is `coaching`. */
    callSidToCoach?: string;
    /** Jitter buffer size for the connecting participant. Twilio will use this setting to apply Jitter Buffer before participant\\\'s audio is mixed into the conference. Can be: `off`, `small`, `medium`, and `large`. Default to `large`. */
    jitterBufferSize?: string;
    /** The SID of a BYOC (Bring Your Own Carrier) trunk to route this call with. Note that `byoc` is only meaningful when `to` is a phone number; it will otherwise be ignored. (Beta) */
    byoc?: string;
    /** The phone number, Client identifier, or username portion of SIP address that made this call. Phone numbers are in [E.164](https://www.twilio.com/docs/glossary/what-e164) format (e.g., +16175551212). Client identifiers are formatted `client:name`. If using a phone number, it must be a Twilio number or a Verified [outgoing caller id](https://www.twilio.com/docs/voice/api/outgoing-caller-ids) for your account. If the `to` parameter is a phone number, `callerId` must also be a phone number. If `to` is sip address, this value of `callerId` should be a username portion to be used to populate the From header that is passed to the SIP endpoint. */
    callerId?: string;
    /** The Reason for the outgoing call. Use it to specify the purpose of the call that is presented on the called party\\\'s phone. (Branded Calls Beta) */
    callReason?: string;
    /** The audio track to record for the call. Can be: `inbound`, `outbound` or `both`. The default is `both`. `inbound` records the audio that is received by Twilio. `outbound` records the audio that is sent from Twilio. `both` records the audio that is received and sent by Twilio. */
    recordingTrack?: string;
    /** The maximum duration of the call in seconds. Constraints depend on account and configuration. */
    timeLimit?: number;
    /** Whether to detect if a human, answering machine, or fax has picked up the call. Can be: `Enable` or `DetectMessageEnd`. Use `Enable` if you would like us to return `AnsweredBy` as soon as the called party is identified. Use `DetectMessageEnd`, if you would like to leave a message on an answering machine. For more information, see [Answering Machine Detection](https://www.twilio.com/docs/voice/answering-machine-detection). */
    machineDetection?: string;
    /** The number of seconds that we should attempt to detect an answering machine before timing out and sending a voice request with `AnsweredBy` of `unknown`. The default timeout is 30 seconds. */
    machineDetectionTimeout?: number;
    /** The number of milliseconds that is used as the measuring stick for the length of the speech activity, where durations lower than this value will be interpreted as a human and longer than this value as a machine. Possible Values: 1000-6000. Default: 2400. */
    machineDetectionSpeechThreshold?: number;
    /** The number of milliseconds of silence after speech activity at which point the speech activity is considered complete. Possible Values: 500-5000. Default: 1200. */
    machineDetectionSpeechEndThreshold?: number;
    /** The number of milliseconds of initial silence after which an `unknown` AnsweredBy result will be returned. Possible Values: 2000-10000. Default: 5000. */
    machineDetectionSilenceTimeout?: number;
    /** The URL that we should call using the `amd_status_callback_method` to notify customer application whether the call was answered by human, machine or fax. */
    amdStatusCallback?: string;
    /** The HTTP method we should use when calling the `amd_status_callback` URL. Can be: `GET` or `POST` and the default is `POST`. */
    amdStatusCallbackMethod?: string;
    /** Whether to trim any leading and trailing silence from the participant recording. Can be: `trim-silence` or `do-not-trim` and the default is `trim-silence`. */
    trim?: string;
    /** A token string needed to invoke a forwarded call. A call_token is generated when an incoming call is received on a Twilio number. Pass an incoming call\\\'s call_token value to a forwarded call via the call_token parameter when creating a new call. A forwarded call should bear the same CallerID of the original incoming call. */
    callToken?: string;
}
/**
 * Options to pass to each
 */
export interface ParticipantListInstanceEachOptions {
    /** Whether to return only participants that are muted. Can be: `true` or `false`. */
    muted?: boolean;
    /** Whether to return only participants that are on hold. Can be: `true` or `false`. */
    hold?: boolean;
    /** Whether to return only participants who are coaching another call. Can be: `true` or `false`. */
    coaching?: boolean;
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
    /** Whether to return only participants that are muted. Can be: `true` or `false`. */
    muted?: boolean;
    /** Whether to return only participants that are on hold. Can be: `true` or `false`. */
    hold?: boolean;
    /** Whether to return only participants who are coaching another call. Can be: `true` or `false`. */
    coaching?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ParticipantListInstancePageOptions {
    /** Whether to return only participants that are muted. Can be: `true` or `false`. */
    muted?: boolean;
    /** Whether to return only participants that are on hold. Can be: `true` or `false`. */
    hold?: boolean;
    /** Whether to return only participants who are coaching another call. Can be: `true` or `false`. */
    coaching?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ParticipantContext {
    /**
     * Remove a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
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
    accountSid: string;
    conferenceSid: string;
    callSid: string;
}
export declare class ParticipantContextImpl implements ParticipantContext {
    protected _version: V2010;
    protected _solution: ParticipantContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, conferenceSid: string, callSid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
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
    account_sid: string;
    call_sid: string;
    label: string;
    call_sid_to_coach: string;
    coaching: boolean;
    conference_sid: string;
    date_created: Date;
    date_updated: Date;
    end_conference_on_exit: boolean;
    muted: boolean;
    hold: boolean;
    start_conference_on_enter: boolean;
    status: ParticipantStatus;
    uri: string;
}
export declare class ParticipantInstance {
    protected _version: V2010;
    protected _solution: ParticipantContextSolution;
    protected _context?: ParticipantContext;
    constructor(_version: V2010, payload: ParticipantResource, accountSid: string, conferenceSid: string, callSid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Participant resource.
     */
    accountSid: string;
    /**
     * The SID of the [Call](https://www.twilio.com/docs/voice/api/call-resource) the Participant resource is associated with.
     */
    callSid: string;
    /**
     * The user-specified label of this participant, if one was given when the participant was created. This may be used to fetch, update or delete the participant.
     */
    label: string;
    /**
     * The SID of the participant who is being `coached`. The participant being coached is the only participant who can hear the participant who is `coaching`.
     */
    callSidToCoach: string;
    /**
     * Whether the participant is coaching another call. Can be: `true` or `false`. If not present, defaults to `false` unless `call_sid_to_coach` is defined. If `true`, `call_sid_to_coach` must be defined.
     */
    coaching: boolean;
    /**
     * The SID of the conference the participant is in.
     */
    conferenceSid: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * Whether the conference ends when the participant leaves. Can be: `true` or `false` and the default is `false`. If `true`, the conference ends and all other participants drop out when the participant leaves.
     */
    endConferenceOnExit: boolean;
    /**
     * Whether the participant is muted. Can be `true` or `false`.
     */
    muted: boolean;
    /**
     * Whether the participant is on hold. Can be `true` or `false`.
     */
    hold: boolean;
    /**
     * Whether the conference starts when the participant joins the conference, if it has not already started. Can be: `true` or `false` and the default is `true`. If `false` and the conference has not started, the participant is muted and hears background music until another participant starts the conference.
     */
    startConferenceOnEnter: boolean;
    status: ParticipantStatus;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Remove a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
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
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        callSid: string;
        label: string;
        callSidToCoach: string;
        coaching: boolean;
        conferenceSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        endConferenceOnExit: boolean;
        muted: boolean;
        hold: boolean;
        startConferenceOnEnter: boolean;
        status: ParticipantStatus;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ParticipantSolution {
    accountSid: string;
    conferenceSid: string;
}
export interface ParticipantListInstance {
    _version: V2010;
    _solution: ParticipantSolution;
    _uri: string;
    (callSid: string): ParticipantContext;
    get(callSid: string): ParticipantContext;
    /**
     * Create a ParticipantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    create(params: ParticipantListInstanceCreateOptions, callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
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
export declare function ParticipantListInstance(version: V2010, accountSid: string, conferenceSid: string): ParticipantListInstance;
export declare class ParticipantPage extends Page<V2010, ParticipantPayload, ParticipantResource, ParticipantInstance> {
    /**
     * Initialize the ParticipantPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: ParticipantSolution);
    /**
     * Build an instance of ParticipantInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ParticipantResource): ParticipantInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
