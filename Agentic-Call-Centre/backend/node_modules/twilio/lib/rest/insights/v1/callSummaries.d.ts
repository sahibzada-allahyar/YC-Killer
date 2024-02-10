/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type CallSummariesAnsweredBy = "unknown" | "machine_start" | "machine_end_beep" | "machine_end_silence" | "machine_end_other" | "human" | "fax";
export type CallSummariesCallState = "ringing" | "completed" | "busy" | "fail" | "noanswer" | "canceled" | "answered" | "undialed";
export type CallSummariesCallType = "carrier" | "sip" | "trunking" | "client";
export type CallSummariesProcessingState = "complete" | "partial";
export type CallSummariesProcessingStateRequest = "completed" | "started" | "partial" | "all";
export type CallSummariesSortBy = "start_time" | "end_time";
/**
 * Options to pass to each
 */
export interface CallSummariesListInstanceEachOptions {
    /** A calling party. Could be an E.164 number, a SIP URI, or a Twilio Client registered name. */
    from?: string;
    /** A called party. Could be an E.164 number, a SIP URI, or a Twilio Client registered name. */
    to?: string;
    /** An origination carrier. */
    fromCarrier?: string;
    /** A destination carrier. */
    toCarrier?: string;
    /** A source country code based on phone number in From. */
    fromCountryCode?: string;
    /** A destination country code. Based on phone number in To. */
    toCountryCode?: string;
    /** A boolean flag indicating whether or not the calls were branded using Twilio Branded Calls. */
    branded?: boolean;
    /** A boolean flag indicating whether or not the caller was verified using SHAKEN/STIR. */
    verifiedCaller?: boolean;
    /** A boolean flag indicating the presence of one or more [Voice Insights Call Tags](https://www.twilio.com/docs/voice/voice-insights/api/call/details-call-tags). */
    hasTag?: boolean;
    /** A Start time of the calls. xm (x minutes), xh (x hours), xd (x days), 1w, 30m, 3d, 4w or datetime-ISO. Defaults to 4h. */
    startTime?: string;
    /** An End Time of the calls. xm (x minutes), xh (x hours), xd (x days), 1w, 30m, 3d, 4w or datetime-ISO. Defaults to 0m. */
    endTime?: string;
    /** A Call Type of the calls. One of `carrier`, `sip`, `trunking` or `client`. */
    callType?: string;
    /** A Call State of the calls. One of `ringing`, `completed`, `busy`, `fail`, `noanswer`, `canceled`, `answered`, `undialed`. */
    callState?: string;
    /** A Direction of the calls. One of `outbound_api`, `outbound_dial`, `inbound`, `trunking_originating`, `trunking_terminating`. */
    direction?: string;
    /** A Processing State of the Call Summaries. One of `completed`, `partial` or `all`. */
    processingState?: CallSummariesProcessingStateRequest;
    /** A Sort By criterion for the returned list of Call Summaries. One of `start_time` or `end_time`. */
    sortBy?: CallSummariesSortBy;
    /** A unique SID identifier of a Subaccount. */
    subaccount?: string;
    /** A boolean flag indicating an abnormal session where the last SIP response was not 200 OK. */
    abnormalSession?: boolean;
    /** An Answered By value for the calls based on `Answering Machine Detection (AMD)`. One of `unknown`, `machine_start`, `machine_end_beep`, `machine_end_silence`, `machine_end_other`, `human` or `fax`. */
    answeredBy?: CallSummariesAnsweredBy;
    /** Either machine or human. */
    answeredByAnnotation?: string;
    /** A Connectivity Issue with the calls. One of `no_connectivity_issue`, `invalid_number`, `caller_id`, `dropped_call`, or `number_reachability`. */
    connectivityIssueAnnotation?: string;
    /** A subjective Quality Issue with the calls. One of `no_quality_issue`, `low_volume`, `choppy_robotic`, `echo`, `dtmf`, `latency`, `owa`, `static_noise`. */
    qualityIssueAnnotation?: string;
    /** A boolean flag indicating spam calls. */
    spamAnnotation?: boolean;
    /** A Call Score of the calls. Use a range of 1-5 to indicate the call experience score, with the following mapping as a reference for the rated call [5: Excellent, 4: Good, 3 : Fair, 2 : Poor, 1: Bad]. */
    callScoreAnnotation?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: CallSummariesInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface CallSummariesListInstanceOptions {
    /** A calling party. Could be an E.164 number, a SIP URI, or a Twilio Client registered name. */
    from?: string;
    /** A called party. Could be an E.164 number, a SIP URI, or a Twilio Client registered name. */
    to?: string;
    /** An origination carrier. */
    fromCarrier?: string;
    /** A destination carrier. */
    toCarrier?: string;
    /** A source country code based on phone number in From. */
    fromCountryCode?: string;
    /** A destination country code. Based on phone number in To. */
    toCountryCode?: string;
    /** A boolean flag indicating whether or not the calls were branded using Twilio Branded Calls. */
    branded?: boolean;
    /** A boolean flag indicating whether or not the caller was verified using SHAKEN/STIR. */
    verifiedCaller?: boolean;
    /** A boolean flag indicating the presence of one or more [Voice Insights Call Tags](https://www.twilio.com/docs/voice/voice-insights/api/call/details-call-tags). */
    hasTag?: boolean;
    /** A Start time of the calls. xm (x minutes), xh (x hours), xd (x days), 1w, 30m, 3d, 4w or datetime-ISO. Defaults to 4h. */
    startTime?: string;
    /** An End Time of the calls. xm (x minutes), xh (x hours), xd (x days), 1w, 30m, 3d, 4w or datetime-ISO. Defaults to 0m. */
    endTime?: string;
    /** A Call Type of the calls. One of `carrier`, `sip`, `trunking` or `client`. */
    callType?: string;
    /** A Call State of the calls. One of `ringing`, `completed`, `busy`, `fail`, `noanswer`, `canceled`, `answered`, `undialed`. */
    callState?: string;
    /** A Direction of the calls. One of `outbound_api`, `outbound_dial`, `inbound`, `trunking_originating`, `trunking_terminating`. */
    direction?: string;
    /** A Processing State of the Call Summaries. One of `completed`, `partial` or `all`. */
    processingState?: CallSummariesProcessingStateRequest;
    /** A Sort By criterion for the returned list of Call Summaries. One of `start_time` or `end_time`. */
    sortBy?: CallSummariesSortBy;
    /** A unique SID identifier of a Subaccount. */
    subaccount?: string;
    /** A boolean flag indicating an abnormal session where the last SIP response was not 200 OK. */
    abnormalSession?: boolean;
    /** An Answered By value for the calls based on `Answering Machine Detection (AMD)`. One of `unknown`, `machine_start`, `machine_end_beep`, `machine_end_silence`, `machine_end_other`, `human` or `fax`. */
    answeredBy?: CallSummariesAnsweredBy;
    /** Either machine or human. */
    answeredByAnnotation?: string;
    /** A Connectivity Issue with the calls. One of `no_connectivity_issue`, `invalid_number`, `caller_id`, `dropped_call`, or `number_reachability`. */
    connectivityIssueAnnotation?: string;
    /** A subjective Quality Issue with the calls. One of `no_quality_issue`, `low_volume`, `choppy_robotic`, `echo`, `dtmf`, `latency`, `owa`, `static_noise`. */
    qualityIssueAnnotation?: string;
    /** A boolean flag indicating spam calls. */
    spamAnnotation?: boolean;
    /** A Call Score of the calls. Use a range of 1-5 to indicate the call experience score, with the following mapping as a reference for the rated call [5: Excellent, 4: Good, 3 : Fair, 2 : Poor, 1: Bad]. */
    callScoreAnnotation?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CallSummariesListInstancePageOptions {
    /** A calling party. Could be an E.164 number, a SIP URI, or a Twilio Client registered name. */
    from?: string;
    /** A called party. Could be an E.164 number, a SIP URI, or a Twilio Client registered name. */
    to?: string;
    /** An origination carrier. */
    fromCarrier?: string;
    /** A destination carrier. */
    toCarrier?: string;
    /** A source country code based on phone number in From. */
    fromCountryCode?: string;
    /** A destination country code. Based on phone number in To. */
    toCountryCode?: string;
    /** A boolean flag indicating whether or not the calls were branded using Twilio Branded Calls. */
    branded?: boolean;
    /** A boolean flag indicating whether or not the caller was verified using SHAKEN/STIR. */
    verifiedCaller?: boolean;
    /** A boolean flag indicating the presence of one or more [Voice Insights Call Tags](https://www.twilio.com/docs/voice/voice-insights/api/call/details-call-tags). */
    hasTag?: boolean;
    /** A Start time of the calls. xm (x minutes), xh (x hours), xd (x days), 1w, 30m, 3d, 4w or datetime-ISO. Defaults to 4h. */
    startTime?: string;
    /** An End Time of the calls. xm (x minutes), xh (x hours), xd (x days), 1w, 30m, 3d, 4w or datetime-ISO. Defaults to 0m. */
    endTime?: string;
    /** A Call Type of the calls. One of `carrier`, `sip`, `trunking` or `client`. */
    callType?: string;
    /** A Call State of the calls. One of `ringing`, `completed`, `busy`, `fail`, `noanswer`, `canceled`, `answered`, `undialed`. */
    callState?: string;
    /** A Direction of the calls. One of `outbound_api`, `outbound_dial`, `inbound`, `trunking_originating`, `trunking_terminating`. */
    direction?: string;
    /** A Processing State of the Call Summaries. One of `completed`, `partial` or `all`. */
    processingState?: CallSummariesProcessingStateRequest;
    /** A Sort By criterion for the returned list of Call Summaries. One of `start_time` or `end_time`. */
    sortBy?: CallSummariesSortBy;
    /** A unique SID identifier of a Subaccount. */
    subaccount?: string;
    /** A boolean flag indicating an abnormal session where the last SIP response was not 200 OK. */
    abnormalSession?: boolean;
    /** An Answered By value for the calls based on `Answering Machine Detection (AMD)`. One of `unknown`, `machine_start`, `machine_end_beep`, `machine_end_silence`, `machine_end_other`, `human` or `fax`. */
    answeredBy?: CallSummariesAnsweredBy;
    /** Either machine or human. */
    answeredByAnnotation?: string;
    /** A Connectivity Issue with the calls. One of `no_connectivity_issue`, `invalid_number`, `caller_id`, `dropped_call`, or `number_reachability`. */
    connectivityIssueAnnotation?: string;
    /** A subjective Quality Issue with the calls. One of `no_quality_issue`, `low_volume`, `choppy_robotic`, `echo`, `dtmf`, `latency`, `owa`, `static_noise`. */
    qualityIssueAnnotation?: string;
    /** A boolean flag indicating spam calls. */
    spamAnnotation?: boolean;
    /** A Call Score of the calls. Use a range of 1-5 to indicate the call experience score, with the following mapping as a reference for the rated call [5: Excellent, 4: Good, 3 : Fair, 2 : Poor, 1: Bad]. */
    callScoreAnnotation?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CallSummariesSolution {
}
export interface CallSummariesListInstance {
    _version: V1;
    _solution: CallSummariesSolution;
    _uri: string;
    /**
     * Streams CallSummariesInstance records from the API.
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
     * @param { CallSummariesListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: CallSummariesInstance, done: (err?: Error) => void) => void): void;
    each(params: CallSummariesListInstanceEachOptions, callback?: (item: CallSummariesInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of CallSummariesInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: CallSummariesPage) => any): Promise<CallSummariesPage>;
    /**
     * Lists CallSummariesInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CallSummariesListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: CallSummariesInstance[]) => any): Promise<CallSummariesInstance[]>;
    list(params: CallSummariesListInstanceOptions, callback?: (error: Error | null, items: CallSummariesInstance[]) => any): Promise<CallSummariesInstance[]>;
    /**
     * Retrieve a single page of CallSummariesInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CallSummariesListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: CallSummariesPage) => any): Promise<CallSummariesPage>;
    page(params: CallSummariesListInstancePageOptions, callback?: (error: Error | null, items: CallSummariesPage) => any): Promise<CallSummariesPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CallSummariesListInstance(version: V1): CallSummariesListInstance;
interface CallSummariesPayload extends TwilioResponsePayload {
    call_summaries: CallSummariesResource[];
}
interface CallSummariesResource {
    account_sid: string;
    call_sid: string;
    answered_by: CallSummariesAnsweredBy;
    call_type: CallSummariesCallType;
    call_state: CallSummariesCallState;
    processing_state: CallSummariesProcessingState;
    created_time: Date;
    start_time: Date;
    end_time: Date;
    duration: number;
    connect_duration: number;
    from: any;
    to: any;
    carrier_edge: any;
    client_edge: any;
    sdk_edge: any;
    sip_edge: any;
    tags: Array<string>;
    url: string;
    attributes: any;
    properties: any;
    trust: any;
    annotation: any;
}
export declare class CallSummariesInstance {
    protected _version: V1;
    constructor(_version: V1, payload: CallSummariesResource);
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * The unique SID identifier of the Call.
     */
    callSid: string;
    answeredBy: CallSummariesAnsweredBy;
    callType: CallSummariesCallType;
    callState: CallSummariesCallState;
    processingState: CallSummariesProcessingState;
    /**
     * The time at which the Call was created, given in ISO 8601 format. Can be different from `start_time` in the event of queueing due to CPS
     */
    createdTime: Date;
    /**
     * The time at which the Call was started, given in ISO 8601 format.
     */
    startTime: Date;
    /**
     * The time at which the Call was ended, given in ISO 8601 format.
     */
    endTime: Date;
    /**
     * Duration between when the call was initiated and the call was ended
     */
    duration: number;
    /**
     * Duration between when the call was answered and when it ended
     */
    connectDuration: number;
    /**
     * The calling party.
     */
    from: any;
    /**
     * The called party.
     */
    to: any;
    /**
     * Contains metrics and properties for the Twilio media gateway of a PSTN call.
     */
    carrierEdge: any;
    /**
     * Contains metrics and properties for the Twilio media gateway of a Client call.
     */
    clientEdge: any;
    /**
     * Contains metrics and properties for the SDK sensor library for Client calls.
     */
    sdkEdge: any;
    /**
     * Contains metrics and properties for the Twilio media gateway of a SIP Interface or Trunking call.
     */
    sipEdge: any;
    /**
     * Tags applied to calls by Voice Insights analysis indicating a condition that could result in subjective degradation of the call quality.
     */
    tags: Array<string>;
    /**
     * The URL of this resource.
     */
    url: string;
    /**
     * Attributes capturing call-flow-specific details.
     */
    attributes: any;
    /**
     * Contains edge-agnostic call-level details.
     */
    properties: any;
    /**
     * Contains trusted communications details including Branded Call and verified caller ID.
     */
    trust: any;
    annotation: any;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        callSid: string;
        answeredBy: CallSummariesAnsweredBy;
        callType: CallSummariesCallType;
        callState: CallSummariesCallState;
        processingState: CallSummariesProcessingState;
        createdTime: Date;
        startTime: Date;
        endTime: Date;
        duration: number;
        connectDuration: number;
        from: any;
        to: any;
        carrierEdge: any;
        clientEdge: any;
        sdkEdge: any;
        sipEdge: any;
        tags: string[];
        url: string;
        attributes: any;
        properties: any;
        trust: any;
        annotation: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class CallSummariesPage extends Page<V1, CallSummariesPayload, CallSummariesResource, CallSummariesInstance> {
    /**
     * Initialize the CallSummariesPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: CallSummariesSolution);
    /**
     * Build an instance of CallSummariesInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CallSummariesResource): CallSummariesInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
