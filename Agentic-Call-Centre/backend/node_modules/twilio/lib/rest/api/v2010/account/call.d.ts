/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
import { EventListInstance } from "./call/event";
import { NotificationListInstance } from "./call/notification";
import { PaymentListInstance } from "./call/payment";
import { RecordingListInstance } from "./call/recording";
import { SiprecListInstance } from "./call/siprec";
import { StreamListInstance } from "./call/stream";
import { UserDefinedMessageListInstance } from "./call/userDefinedMessage";
import { UserDefinedMessageSubscriptionListInstance } from "./call/userDefinedMessageSubscription";
import TwiML from "../../../../twiml/TwiML";
export type CallStatus = "queued" | "ringing" | "in-progress" | "completed" | "busy" | "failed" | "no-answer" | "canceled";
export type CallUpdateStatus = "canceled" | "completed";
/**
 * Options to pass to update a CallInstance
 */
export interface CallContextUpdateOptions {
    /** The absolute URL that returns the TwiML instructions for the call. We will call this URL using the `method` when the call connects. For more information, see the [Url Parameter](https://www.twilio.com/docs/voice/make-calls#specify-a-url-parameter) section in [Making Calls](https://www.twilio.com/docs/voice/make-calls). */
    url?: string;
    /** The HTTP method we should use when calling the `url`. Can be: `GET` or `POST` and the default is `POST`. If an `application_sid` parameter is present, this parameter is ignored. */
    method?: string;
    /**  */
    status?: CallUpdateStatus;
    /** The URL that we call using the `fallback_method` if an error occurs when requesting or executing the TwiML at `url`. If an `application_sid` parameter is present, this parameter is ignored. */
    fallbackUrl?: string;
    /** The HTTP method that we should use to request the `fallback_url`. Can be: `GET` or `POST` and the default is `POST`. If an `application_sid` parameter is present, this parameter is ignored. */
    fallbackMethod?: string;
    /** The URL we should call using the `status_callback_method` to send status information to your application. If no `status_callback_event` is specified, we will send the `completed` status. If an `application_sid` parameter is present, this parameter is ignored. URLs must contain a valid hostname (underscores are not permitted). */
    statusCallback?: string;
    /** The HTTP method we should use when requesting the `status_callback` URL. Can be: `GET` or `POST` and the default is `POST`. If an `application_sid` parameter is present, this parameter is ignored. */
    statusCallbackMethod?: string;
    /** TwiML instructions for the call Twilio will use without fetching Twiml from url. Twiml and url parameters are mutually exclusive */
    twiml?: TwiML | string;
    /** The maximum duration of the call in seconds. Constraints depend on account and configuration. */
    timeLimit?: number;
}
/**
 * Options to pass to create a CallInstance
 */
export interface CallListInstanceCreateOptions {
    /** The phone number, SIP address, or client identifier to call. */
    to: string;
    /** The phone number or client identifier to use as the caller id. If using a phone number, it must be a Twilio number or a Verified [outgoing caller id](https://www.twilio.com/docs/voice/api/outgoing-caller-ids) for your account. If the `to` parameter is a phone number, `From` must also be a phone number. */
    from: string;
    /** The HTTP method we should use when calling the `url` parameter\\\'s value. Can be: `GET` or `POST` and the default is `POST`. If an `application_sid` parameter is present, this parameter is ignored. */
    method?: string;
    /** The URL that we call using the `fallback_method` if an error occurs when requesting or executing the TwiML at `url`. If an `application_sid` parameter is present, this parameter is ignored. */
    fallbackUrl?: string;
    /** The HTTP method that we should use to request the `fallback_url`. Can be: `GET` or `POST` and the default is `POST`. If an `application_sid` parameter is present, this parameter is ignored. */
    fallbackMethod?: string;
    /** The URL we should call using the `status_callback_method` to send status information to your application. If no `status_callback_event` is specified, we will send the `completed` status. If an `application_sid` parameter is present, this parameter is ignored. URLs must contain a valid hostname (underscores are not permitted). */
    statusCallback?: string;
    /** The call progress events that we will send to the `status_callback` URL. Can be: `initiated`, `ringing`, `answered`, and `completed`. If no event is specified, we send the `completed` status. If you want to receive multiple events, specify each one in a separate `status_callback_event` parameter. See the code sample for [monitoring call progress](https://www.twilio.com/docs/voice/api/call-resource?code-sample=code-create-a-call-resource-and-specify-a-statuscallbackevent&code-sdk-version=json). If an `application_sid` is present, this parameter is ignored. */
    statusCallbackEvent?: Array<string>;
    /** The HTTP method we should use when calling the `status_callback` URL. Can be: `GET` or `POST` and the default is `POST`. If an `application_sid` parameter is present, this parameter is ignored. */
    statusCallbackMethod?: string;
    /** A string of keys to dial after connecting to the number, maximum of 32 digits. Valid digits in the string include: any digit (`0`-`9`), \\\'`#`\\\', \\\'`*`\\\' and \\\'`w`\\\', to insert a half second pause. For example, if you connected to a company phone number and wanted to pause for one second, and then dial extension 1234 followed by the pound key, the value of this parameter would be `ww1234#`. Remember to URL-encode this string, since the \\\'`#`\\\' character has special meaning in a URL. If both `SendDigits` and `MachineDetection` parameters are provided, then `MachineDetection` will be ignored. */
    sendDigits?: string;
    /** The integer number of seconds that we should allow the phone to ring before assuming there is no answer. The default is `60` seconds and the maximum is `600` seconds. For some call flows, we will add a 5-second buffer to the timeout value you provide. For this reason, a timeout value of 10 seconds could result in an actual timeout closer to 15 seconds. You can set this to a short time, such as `15` seconds, to hang up before reaching an answering machine or voicemail. */
    timeout?: number;
    /** Whether to record the call. Can be `true` to record the phone call, or `false` to not. The default is `false`. The `recording_url` is sent to the `status_callback` URL. */
    record?: boolean;
    /** The number of channels in the final recording. Can be: `mono` or `dual`. The default is `mono`. `mono` records both legs of the call in a single channel of the recording file. `dual` records each leg to a separate channel of the recording file. The first channel of a dual-channel recording contains the parent call and the second channel contains the child call. */
    recordingChannels?: string;
    /** The URL that we call when the recording is available to be accessed. */
    recordingStatusCallback?: string;
    /** The HTTP method we should use when calling the `recording_status_callback` URL. Can be: `GET` or `POST` and the default is `POST`. */
    recordingStatusCallbackMethod?: string;
    /** The username used to authenticate the caller making a SIP call. */
    sipAuthUsername?: string;
    /** The password required to authenticate the user account specified in `sip_auth_username`. */
    sipAuthPassword?: string;
    /** Whether to detect if a human, answering machine, or fax has picked up the call. Can be: `Enable` or `DetectMessageEnd`. Use `Enable` if you would like us to return `AnsweredBy` as soon as the called party is identified. Use `DetectMessageEnd`, if you would like to leave a message on an answering machine. If `send_digits` is provided, this parameter is ignored. For more information, see [Answering Machine Detection](https://www.twilio.com/docs/voice/answering-machine-detection). */
    machineDetection?: string;
    /** The number of seconds that we should attempt to detect an answering machine before timing out and sending a voice request with `AnsweredBy` of `unknown`. The default timeout is 30 seconds. */
    machineDetectionTimeout?: number;
    /** The recording status events that will trigger calls to the URL specified in `recording_status_callback`. Can be: `in-progress`, `completed` and `absent`. Defaults to `completed`. Separate  multiple values with a space. */
    recordingStatusCallbackEvent?: Array<string>;
    /** Whether to trim any leading and trailing silence from the recording. Can be: `trim-silence` or `do-not-trim` and the default is `trim-silence`. */
    trim?: string;
    /** The phone number, SIP address, or Client identifier that made this call. Phone numbers are in [E.164 format](https://wwnw.twilio.com/docs/glossary/what-e164) (e.g., +16175551212). SIP addresses are formatted as `name@company.com`. */
    callerId?: string;
    /** The number of milliseconds that is used as the measuring stick for the length of the speech activity, where durations lower than this value will be interpreted as a human and longer than this value as a machine. Possible Values: 1000-6000. Default: 2400. */
    machineDetectionSpeechThreshold?: number;
    /** The number of milliseconds of silence after speech activity at which point the speech activity is considered complete. Possible Values: 500-5000. Default: 1200. */
    machineDetectionSpeechEndThreshold?: number;
    /** The number of milliseconds of initial silence after which an `unknown` AnsweredBy result will be returned. Possible Values: 2000-10000. Default: 5000. */
    machineDetectionSilenceTimeout?: number;
    /** Select whether to perform answering machine detection in the background. Default, blocks the execution of the call until Answering Machine Detection is completed. Can be: `true` or `false`. */
    asyncAmd?: string;
    /** The URL that we should call using the `async_amd_status_callback_method` to notify customer application whether the call was answered by human, machine or fax. */
    asyncAmdStatusCallback?: string;
    /** The HTTP method we should use when calling the `async_amd_status_callback` URL. Can be: `GET` or `POST` and the default is `POST`. */
    asyncAmdStatusCallbackMethod?: string;
    /** The SID of a BYOC (Bring Your Own Carrier) trunk to route this call with. Note that `byoc` is only meaningful when `to` is a phone number; it will otherwise be ignored. (Beta) */
    byoc?: string;
    /** The Reason for the outgoing call. Use it to specify the purpose of the call that is presented on the called party\\\'s phone. (Branded Calls Beta) */
    callReason?: string;
    /** A token string needed to invoke a forwarded call. A call_token is generated when an incoming call is received on a Twilio number. Pass an incoming call\\\'s call_token value to a forwarded call via the call_token parameter when creating a new call. A forwarded call should bear the same CallerID of the original incoming call. */
    callToken?: string;
    /** The audio track to record for the call. Can be: `inbound`, `outbound` or `both`. The default is `both`. `inbound` records the audio that is received by Twilio. `outbound` records the audio that is generated from Twilio. `both` records the audio that is received and generated by Twilio. */
    recordingTrack?: string;
    /** The maximum duration of the call in seconds. Constraints depend on account and configuration. */
    timeLimit?: number;
    /** The absolute URL that returns the TwiML instructions for the call. We will call this URL using the `method` when the call connects. For more information, see the [Url Parameter](https://www.twilio.com/docs/voice/make-calls#specify-a-url-parameter) section in [Making Calls](https://www.twilio.com/docs/voice/make-calls). */
    url?: string;
    /** TwiML instructions for the call Twilio will use without fetching Twiml from url parameter. If both `twiml` and `url` are provided then `twiml` parameter will be ignored. Max 4000 characters. */
    twiml?: TwiML | string;
    /** The SID of the Application resource that will handle the call, if the call will be handled by an application. */
    applicationSid?: string;
}
/**
 * Options to pass to each
 */
export interface CallListInstanceEachOptions {
    /** Only show calls made to this phone number, SIP address, Client identifier or SIM SID. */
    to?: string;
    /** Only include calls from this phone number, SIP address, Client identifier or SIM SID. */
    from?: string;
    /** Only include calls spawned by calls with this SID. */
    parentCallSid?: string;
    /** The status of the calls to include. Can be: `queued`, `ringing`, `in-progress`, `canceled`, `completed`, `failed`, `busy`, or `no-answer`. */
    status?: CallStatus;
    /** Only include calls that started on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that started on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read calls that started on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read calls that started on or after midnight of this date. */
    startTime?: Date;
    /** Only include calls that started on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that started on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read calls that started on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read calls that started on or after midnight of this date. */
    startTimeBefore?: Date;
    /** Only include calls that started on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that started on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read calls that started on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read calls that started on or after midnight of this date. */
    startTimeAfter?: Date;
    /** Only include calls that ended on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that ended on this date. You can also specify an inequality, such as `EndTime<=YYYY-MM-DD`, to read calls that ended on or before midnight of this date, and `EndTime>=YYYY-MM-DD` to read calls that ended on or after midnight of this date. */
    endTime?: Date;
    /** Only include calls that ended on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that ended on this date. You can also specify an inequality, such as `EndTime<=YYYY-MM-DD`, to read calls that ended on or before midnight of this date, and `EndTime>=YYYY-MM-DD` to read calls that ended on or after midnight of this date. */
    endTimeBefore?: Date;
    /** Only include calls that ended on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that ended on this date. You can also specify an inequality, such as `EndTime<=YYYY-MM-DD`, to read calls that ended on or before midnight of this date, and `EndTime>=YYYY-MM-DD` to read calls that ended on or after midnight of this date. */
    endTimeAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: CallInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface CallListInstanceOptions {
    /** Only show calls made to this phone number, SIP address, Client identifier or SIM SID. */
    to?: string;
    /** Only include calls from this phone number, SIP address, Client identifier or SIM SID. */
    from?: string;
    /** Only include calls spawned by calls with this SID. */
    parentCallSid?: string;
    /** The status of the calls to include. Can be: `queued`, `ringing`, `in-progress`, `canceled`, `completed`, `failed`, `busy`, or `no-answer`. */
    status?: CallStatus;
    /** Only include calls that started on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that started on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read calls that started on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read calls that started on or after midnight of this date. */
    startTime?: Date;
    /** Only include calls that started on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that started on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read calls that started on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read calls that started on or after midnight of this date. */
    startTimeBefore?: Date;
    /** Only include calls that started on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that started on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read calls that started on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read calls that started on or after midnight of this date. */
    startTimeAfter?: Date;
    /** Only include calls that ended on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that ended on this date. You can also specify an inequality, such as `EndTime<=YYYY-MM-DD`, to read calls that ended on or before midnight of this date, and `EndTime>=YYYY-MM-DD` to read calls that ended on or after midnight of this date. */
    endTime?: Date;
    /** Only include calls that ended on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that ended on this date. You can also specify an inequality, such as `EndTime<=YYYY-MM-DD`, to read calls that ended on or before midnight of this date, and `EndTime>=YYYY-MM-DD` to read calls that ended on or after midnight of this date. */
    endTimeBefore?: Date;
    /** Only include calls that ended on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that ended on this date. You can also specify an inequality, such as `EndTime<=YYYY-MM-DD`, to read calls that ended on or before midnight of this date, and `EndTime>=YYYY-MM-DD` to read calls that ended on or after midnight of this date. */
    endTimeAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CallListInstancePageOptions {
    /** Only show calls made to this phone number, SIP address, Client identifier or SIM SID. */
    to?: string;
    /** Only include calls from this phone number, SIP address, Client identifier or SIM SID. */
    from?: string;
    /** Only include calls spawned by calls with this SID. */
    parentCallSid?: string;
    /** The status of the calls to include. Can be: `queued`, `ringing`, `in-progress`, `canceled`, `completed`, `failed`, `busy`, or `no-answer`. */
    status?: CallStatus;
    /** Only include calls that started on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that started on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read calls that started on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read calls that started on or after midnight of this date. */
    startTime?: Date;
    /** Only include calls that started on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that started on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read calls that started on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read calls that started on or after midnight of this date. */
    startTimeBefore?: Date;
    /** Only include calls that started on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that started on this date. You can also specify an inequality, such as `StartTime<=YYYY-MM-DD`, to read calls that started on or before midnight of this date, and `StartTime>=YYYY-MM-DD` to read calls that started on or after midnight of this date. */
    startTimeAfter?: Date;
    /** Only include calls that ended on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that ended on this date. You can also specify an inequality, such as `EndTime<=YYYY-MM-DD`, to read calls that ended on or before midnight of this date, and `EndTime>=YYYY-MM-DD` to read calls that ended on or after midnight of this date. */
    endTime?: Date;
    /** Only include calls that ended on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that ended on this date. You can also specify an inequality, such as `EndTime<=YYYY-MM-DD`, to read calls that ended on or before midnight of this date, and `EndTime>=YYYY-MM-DD` to read calls that ended on or after midnight of this date. */
    endTimeBefore?: Date;
    /** Only include calls that ended on this date. Specify a date as `YYYY-MM-DD` in GMT, for example: `2009-07-06`, to read only calls that ended on this date. You can also specify an inequality, such as `EndTime<=YYYY-MM-DD`, to read calls that ended on or before midnight of this date, and `EndTime>=YYYY-MM-DD` to read calls that ended on or after midnight of this date. */
    endTimeAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CallContext {
    events: EventListInstance;
    notifications: NotificationListInstance;
    payments: PaymentListInstance;
    recordings: RecordingListInstance;
    siprec: SiprecListInstance;
    streams: StreamListInstance;
    userDefinedMessages: UserDefinedMessageListInstance;
    userDefinedMessageSubscriptions: UserDefinedMessageSubscriptionListInstance;
    /**
     * Remove a CallInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CallInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CallInstance
     */
    fetch(callback?: (error: Error | null, item?: CallInstance) => any): Promise<CallInstance>;
    /**
     * Update a CallInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CallInstance
     */
    update(callback?: (error: Error | null, item?: CallInstance) => any): Promise<CallInstance>;
    /**
     * Update a CallInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CallInstance
     */
    update(params: CallContextUpdateOptions, callback?: (error: Error | null, item?: CallInstance) => any): Promise<CallInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface CallContextSolution {
    accountSid: string;
    sid: string;
}
export declare class CallContextImpl implements CallContext {
    protected _version: V2010;
    protected _solution: CallContextSolution;
    protected _uri: string;
    protected _events?: EventListInstance;
    protected _notifications?: NotificationListInstance;
    protected _payments?: PaymentListInstance;
    protected _recordings?: RecordingListInstance;
    protected _siprec?: SiprecListInstance;
    protected _streams?: StreamListInstance;
    protected _userDefinedMessages?: UserDefinedMessageListInstance;
    protected _userDefinedMessageSubscriptions?: UserDefinedMessageSubscriptionListInstance;
    constructor(_version: V2010, accountSid: string, sid: string);
    get events(): EventListInstance;
    get notifications(): NotificationListInstance;
    get payments(): PaymentListInstance;
    get recordings(): RecordingListInstance;
    get siprec(): SiprecListInstance;
    get streams(): StreamListInstance;
    get userDefinedMessages(): UserDefinedMessageListInstance;
    get userDefinedMessageSubscriptions(): UserDefinedMessageSubscriptionListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: CallInstance) => any): Promise<CallInstance>;
    update(params?: CallContextUpdateOptions | ((error: Error | null, item?: CallInstance) => any), callback?: (error: Error | null, item?: CallInstance) => any): Promise<CallInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): CallContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface CallPayload extends TwilioResponsePayload {
    calls: CallResource[];
}
interface CallResource {
    sid: string;
    date_created: Date;
    date_updated: Date;
    parent_call_sid: string;
    account_sid: string;
    to: string;
    to_formatted: string;
    from: string;
    from_formatted: string;
    phone_number_sid: string;
    status: CallStatus;
    start_time: Date;
    end_time: Date;
    duration: string;
    price: string;
    price_unit: string;
    direction: string;
    answered_by: string;
    api_version: string;
    forwarded_from: string;
    group_sid: string;
    caller_name: string;
    queue_time: string;
    trunk_sid: string;
    uri: string;
    subresource_uris: Record<string, string>;
}
export declare class CallInstance {
    protected _version: V2010;
    protected _solution: CallContextSolution;
    protected _context?: CallContext;
    constructor(_version: V2010, payload: CallResource, accountSid: string, sid?: string);
    /**
     * The unique string that we created to identify this Call resource.
     */
    sid: string;
    /**
     * The date and time in GMT that this resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that this resource was last updated, specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The SID that identifies the call that created this leg.
     */
    parentCallSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created this Call resource.
     */
    accountSid: string;
    /**
     * The phone number, SIP address, Client identifier or SIM SID that received this call. Phone numbers are in [E.164](https://www.twilio.com/docs/glossary/what-e164) format (e.g., +16175551212). SIP addresses are formatted as `name@company.com`. Client identifiers are formatted `client:name`. SIM SIDs are formatted as `sim:sid`.
     */
    to: string;
    /**
     * The phone number, SIP address or Client identifier that received this call. Formatted for display. Non-North American phone numbers are in [E.164](https://www.twilio.com/docs/glossary/what-e164) format (e.g., +442071838750).
     */
    toFormatted: string;
    /**
     * The phone number, SIP address, Client identifier or SIM SID that made this call. Phone numbers are in [E.164](https://www.twilio.com/docs/glossary/what-e164) format (e.g., +16175551212). SIP addresses are formatted as `name@company.com`. Client identifiers are formatted `client:name`. SIM SIDs are formatted as `sim:sid`.
     */
    from: string;
    /**
     * The calling phone number, SIP address, or Client identifier formatted for display. Non-North American phone numbers are in [E.164](https://www.twilio.com/docs/glossary/what-e164) format (e.g., +442071838750).
     */
    fromFormatted: string;
    /**
     * If the call was inbound, this is the SID of the IncomingPhoneNumber resource that received the call. If the call was outbound, it is the SID of the OutgoingCallerId resource from which the call was placed.
     */
    phoneNumberSid: string;
    status: CallStatus;
    /**
     * The start time of the call, given as GMT in [RFC 2822](https://www.php.net/manual/en/class.datetime.php#datetime.constants.rfc2822) format. Empty if the call has not yet been dialed.
     */
    startTime: Date;
    /**
     * The time the call ended, given as GMT in [RFC 2822](https://www.php.net/manual/en/class.datetime.php#datetime.constants.rfc2822) format. Empty if the call did not complete successfully.
     */
    endTime: Date;
    /**
     * The length of the call in seconds. This value is empty for busy, failed, unanswered, or ongoing calls.
     */
    duration: string;
    /**
     * The charge for this call, in the currency associated with the account. Populated after the call is completed. May not be immediately available.
     */
    price: string;
    /**
     * The currency in which `Price` is measured, in [ISO 4127](https://www.iso.org/iso/home/standards/currency_codes.htm) format (e.g., `USD`, `EUR`, `JPY`). Always capitalized for calls.
     */
    priceUnit: string;
    /**
     * A string describing the direction of the call. Can be: `inbound` for inbound calls, `outbound-api` for calls initiated via the REST API or `outbound-dial` for calls initiated by a `<Dial>` verb. Using [Elastic SIP Trunking](https://www.twilio.com/docs/sip-trunking), the values can be [`trunking-terminating`](https://www.twilio.com/docs/sip-trunking#termination) for outgoing calls from your communications infrastructure to the PSTN or [`trunking-originating`](https://www.twilio.com/docs/sip-trunking#origination) for incoming calls to your communications infrastructure from the PSTN.
     */
    direction: string;
    /**
     * Either `human` or `machine` if this call was initiated with answering machine detection. Empty otherwise.
     */
    answeredBy: string;
    /**
     * The API version used to create the call.
     */
    apiVersion: string;
    /**
     * The forwarding phone number if this call was an incoming call forwarded from another number (depends on carrier supporting forwarding). Otherwise, empty.
     */
    forwardedFrom: string;
    /**
     * The Group SID associated with this call. If no Group is associated with the call, the field is empty.
     */
    groupSid: string;
    /**
     * The caller\'s name if this call was an incoming call to a phone number with caller ID Lookup enabled. Otherwise, empty.
     */
    callerName: string;
    /**
     * The wait time in milliseconds before the call is placed.
     */
    queueTime: string;
    /**
     * The unique identifier of the trunk resource that was used for this call. The field is empty if the call was not made using a SIP trunk or if the call is not terminated.
     */
    trunkSid: string;
    /**
     * The URI of this resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    /**
     * A list of subresources available to this call, identified by their URIs relative to `https://api.twilio.com`.
     */
    subresourceUris: Record<string, string>;
    private get _proxy();
    /**
     * Remove a CallInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CallInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CallInstance
     */
    fetch(callback?: (error: Error | null, item?: CallInstance) => any): Promise<CallInstance>;
    /**
     * Update a CallInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CallInstance
     */
    update(callback?: (error: Error | null, item?: CallInstance) => any): Promise<CallInstance>;
    /**
     * Update a CallInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CallInstance
     */
    update(params: CallContextUpdateOptions, callback?: (error: Error | null, item?: CallInstance) => any): Promise<CallInstance>;
    /**
     * Access the events.
     */
    events(): EventListInstance;
    /**
     * Access the notifications.
     */
    notifications(): NotificationListInstance;
    /**
     * Access the payments.
     */
    payments(): PaymentListInstance;
    /**
     * Access the recordings.
     */
    recordings(): RecordingListInstance;
    /**
     * Access the siprec.
     */
    siprec(): SiprecListInstance;
    /**
     * Access the streams.
     */
    streams(): StreamListInstance;
    /**
     * Access the userDefinedMessages.
     */
    userDefinedMessages(): UserDefinedMessageListInstance;
    /**
     * Access the userDefinedMessageSubscriptions.
     */
    userDefinedMessageSubscriptions(): UserDefinedMessageSubscriptionListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        dateCreated: Date;
        dateUpdated: Date;
        parentCallSid: string;
        accountSid: string;
        to: string;
        toFormatted: string;
        from: string;
        fromFormatted: string;
        phoneNumberSid: string;
        status: CallStatus;
        startTime: Date;
        endTime: Date;
        duration: string;
        price: string;
        priceUnit: string;
        direction: string;
        answeredBy: string;
        apiVersion: string;
        forwardedFrom: string;
        groupSid: string;
        callerName: string;
        queueTime: string;
        trunkSid: string;
        uri: string;
        subresourceUris: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CallSolution {
    accountSid: string;
}
export interface CallListInstance {
    _version: V2010;
    _solution: CallSolution;
    _uri: string;
    (sid: string): CallContext;
    get(sid: string): CallContext;
    /**
     * Create a CallInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CallInstance
     */
    create(params: CallListInstanceCreateOptions, callback?: (error: Error | null, item?: CallInstance) => any): Promise<CallInstance>;
    /**
     * Streams CallInstance records from the API.
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
     * @param { CallListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: CallInstance, done: (err?: Error) => void) => void): void;
    each(params: CallListInstanceEachOptions, callback?: (item: CallInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of CallInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: CallPage) => any): Promise<CallPage>;
    /**
     * Lists CallInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CallListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: CallInstance[]) => any): Promise<CallInstance[]>;
    list(params: CallListInstanceOptions, callback?: (error: Error | null, items: CallInstance[]) => any): Promise<CallInstance[]>;
    /**
     * Retrieve a single page of CallInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CallListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: CallPage) => any): Promise<CallPage>;
    page(params: CallListInstancePageOptions, callback?: (error: Error | null, items: CallPage) => any): Promise<CallPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CallListInstance(version: V2010, accountSid: string): CallListInstance;
export declare class CallPage extends Page<V2010, CallPayload, CallResource, CallInstance> {
    /**
     * Initialize the CallPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: CallSolution);
    /**
     * Build an instance of CallInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CallResource): CallInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
