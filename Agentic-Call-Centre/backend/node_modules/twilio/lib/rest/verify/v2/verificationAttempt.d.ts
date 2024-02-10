/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V2 from "../V2";
export type VerificationAttemptChannels = "sms" | "call" | "email" | "whatsapp";
export type VerificationAttemptConversionStatus = "converted" | "unconverted";
/**
 * Options to pass to each
 */
export interface VerificationAttemptListInstanceEachOptions {
    /** Datetime filter used to consider only Verification Attempts created after this datetime on the summary aggregation. Given as GMT in ISO 8601 formatted datetime string: yyyy-MM-dd\'T\'HH:mm:ss\'Z. */
    dateCreatedAfter?: Date;
    /** Datetime filter used to consider only Verification Attempts created before this datetime on the summary aggregation. Given as GMT in ISO 8601 formatted datetime string: yyyy-MM-dd\'T\'HH:mm:ss\'Z. */
    dateCreatedBefore?: Date;
    /** Destination of a verification. It is phone number in E.164 format. */
    "channelData.to"?: string;
    /** Filter used to query Verification Attempts sent to the specified destination country. */
    country?: string;
    /** Filter used to query Verification Attempts by communication channel. Valid values are `SMS` and `CALL` */
    channel?: VerificationAttemptChannels;
    /** Filter used to query Verification Attempts by verify service. Only attempts of the provided SID will be returned. */
    verifyServiceSid?: string;
    /** Filter used to return all the Verification Attempts of a single verification. Only attempts of the provided verification SID will be returned. */
    verificationSid?: string;
    /** Filter used to query Verification Attempts by conversion status. Valid values are `UNCONVERTED`, for attempts that were not converted, and `CONVERTED`, for attempts that were confirmed. */
    status?: VerificationAttemptConversionStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: VerificationAttemptInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface VerificationAttemptListInstanceOptions {
    /** Datetime filter used to consider only Verification Attempts created after this datetime on the summary aggregation. Given as GMT in ISO 8601 formatted datetime string: yyyy-MM-dd\'T\'HH:mm:ss\'Z. */
    dateCreatedAfter?: Date;
    /** Datetime filter used to consider only Verification Attempts created before this datetime on the summary aggregation. Given as GMT in ISO 8601 formatted datetime string: yyyy-MM-dd\'T\'HH:mm:ss\'Z. */
    dateCreatedBefore?: Date;
    /** Destination of a verification. It is phone number in E.164 format. */
    "channelData.to"?: string;
    /** Filter used to query Verification Attempts sent to the specified destination country. */
    country?: string;
    /** Filter used to query Verification Attempts by communication channel. Valid values are `SMS` and `CALL` */
    channel?: VerificationAttemptChannels;
    /** Filter used to query Verification Attempts by verify service. Only attempts of the provided SID will be returned. */
    verifyServiceSid?: string;
    /** Filter used to return all the Verification Attempts of a single verification. Only attempts of the provided verification SID will be returned. */
    verificationSid?: string;
    /** Filter used to query Verification Attempts by conversion status. Valid values are `UNCONVERTED`, for attempts that were not converted, and `CONVERTED`, for attempts that were confirmed. */
    status?: VerificationAttemptConversionStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface VerificationAttemptListInstancePageOptions {
    /** Datetime filter used to consider only Verification Attempts created after this datetime on the summary aggregation. Given as GMT in ISO 8601 formatted datetime string: yyyy-MM-dd\'T\'HH:mm:ss\'Z. */
    dateCreatedAfter?: Date;
    /** Datetime filter used to consider only Verification Attempts created before this datetime on the summary aggregation. Given as GMT in ISO 8601 formatted datetime string: yyyy-MM-dd\'T\'HH:mm:ss\'Z. */
    dateCreatedBefore?: Date;
    /** Destination of a verification. It is phone number in E.164 format. */
    "channelData.to"?: string;
    /** Filter used to query Verification Attempts sent to the specified destination country. */
    country?: string;
    /** Filter used to query Verification Attempts by communication channel. Valid values are `SMS` and `CALL` */
    channel?: VerificationAttemptChannels;
    /** Filter used to query Verification Attempts by verify service. Only attempts of the provided SID will be returned. */
    verifyServiceSid?: string;
    /** Filter used to return all the Verification Attempts of a single verification. Only attempts of the provided verification SID will be returned. */
    verificationSid?: string;
    /** Filter used to query Verification Attempts by conversion status. Valid values are `UNCONVERTED`, for attempts that were not converted, and `CONVERTED`, for attempts that were confirmed. */
    status?: VerificationAttemptConversionStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface VerificationAttemptContext {
    /**
     * Fetch a VerificationAttemptInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationAttemptInstance
     */
    fetch(callback?: (error: Error | null, item?: VerificationAttemptInstance) => any): Promise<VerificationAttemptInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface VerificationAttemptContextSolution {
    sid: string;
}
export declare class VerificationAttemptContextImpl implements VerificationAttemptContext {
    protected _version: V2;
    protected _solution: VerificationAttemptContextSolution;
    protected _uri: string;
    constructor(_version: V2, sid: string);
    fetch(callback?: (error: Error | null, item?: VerificationAttemptInstance) => any): Promise<VerificationAttemptInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): VerificationAttemptContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface VerificationAttemptPayload extends TwilioResponsePayload {
    attempts: VerificationAttemptResource[];
}
interface VerificationAttemptResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    verification_sid: string;
    date_created: Date;
    date_updated: Date;
    conversion_status: VerificationAttemptConversionStatus;
    channel: VerificationAttemptChannels;
    price: any;
    channel_data: any;
    url: string;
}
export declare class VerificationAttemptInstance {
    protected _version: V2;
    protected _solution: VerificationAttemptContextSolution;
    protected _context?: VerificationAttemptContext;
    constructor(_version: V2, payload: VerificationAttemptResource, sid?: string);
    /**
     * The SID that uniquely identifies the verification attempt resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Verification resource.
     */
    accountSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/verify/api/service) used to generate the attempt.
     */
    serviceSid: string;
    /**
     * The SID of the [Verification](https://www.twilio.com/docs/verify/api/verification) that generated the attempt.
     */
    verificationSid: string;
    /**
     * The date that this Attempt was created, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date that this Attempt was updated, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    conversionStatus: VerificationAttemptConversionStatus;
    channel: VerificationAttemptChannels;
    /**
     * An object containing the charge for this verification attempt related to the channel costs and the currency used. The costs related to the succeeded verifications are not included. May not be immediately available. More information on pricing is available [here](https://www.twilio.com/en-us/verify/pricing).
     */
    price: any;
    /**
     * An object containing the channel specific information for an attempt.
     */
    channelData: any;
    url: string;
    private get _proxy();
    /**
     * Fetch a VerificationAttemptInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationAttemptInstance
     */
    fetch(callback?: (error: Error | null, item?: VerificationAttemptInstance) => any): Promise<VerificationAttemptInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        verificationSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        conversionStatus: VerificationAttemptConversionStatus;
        channel: VerificationAttemptChannels;
        price: any;
        channelData: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface VerificationAttemptSolution {
}
export interface VerificationAttemptListInstance {
    _version: V2;
    _solution: VerificationAttemptSolution;
    _uri: string;
    (sid: string): VerificationAttemptContext;
    get(sid: string): VerificationAttemptContext;
    /**
     * Streams VerificationAttemptInstance records from the API.
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
     * @param { VerificationAttemptListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: VerificationAttemptInstance, done: (err?: Error) => void) => void): void;
    each(params: VerificationAttemptListInstanceEachOptions, callback?: (item: VerificationAttemptInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of VerificationAttemptInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: VerificationAttemptPage) => any): Promise<VerificationAttemptPage>;
    /**
     * Lists VerificationAttemptInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { VerificationAttemptListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: VerificationAttemptInstance[]) => any): Promise<VerificationAttemptInstance[]>;
    list(params: VerificationAttemptListInstanceOptions, callback?: (error: Error | null, items: VerificationAttemptInstance[]) => any): Promise<VerificationAttemptInstance[]>;
    /**
     * Retrieve a single page of VerificationAttemptInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { VerificationAttemptListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: VerificationAttemptPage) => any): Promise<VerificationAttemptPage>;
    page(params: VerificationAttemptListInstancePageOptions, callback?: (error: Error | null, items: VerificationAttemptPage) => any): Promise<VerificationAttemptPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function VerificationAttemptListInstance(version: V2): VerificationAttemptListInstance;
export declare class VerificationAttemptPage extends Page<V2, VerificationAttemptPayload, VerificationAttemptResource, VerificationAttemptInstance> {
    /**
     * Initialize the VerificationAttemptPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: VerificationAttemptSolution);
    /**
     * Build an instance of VerificationAttemptInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: VerificationAttemptResource): VerificationAttemptInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
