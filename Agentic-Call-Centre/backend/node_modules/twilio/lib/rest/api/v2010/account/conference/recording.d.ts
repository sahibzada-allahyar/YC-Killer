/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2010 from "../../../V2010";
export type RecordingSource = "DialVerb" | "Conference" | "OutboundAPI" | "Trunking" | "RecordVerb" | "StartCallRecordingAPI" | "StartConferenceRecordingAPI";
export type RecordingStatus = "in-progress" | "paused" | "stopped" | "processing" | "completed" | "absent";
/**
 * Options to pass to update a RecordingInstance
 */
export interface RecordingContextUpdateOptions {
    /**  */
    status: RecordingStatus;
    /** Whether to record during a pause. Can be: `skip` or `silence` and the default is `silence`. `skip` does not record during the pause period, while `silence` will replace the actual audio of the call with silence during the pause period. This parameter only applies when setting `status` is set to `paused`. */
    pauseBehavior?: string;
}
/**
 * Options to pass to each
 */
export interface RecordingListInstanceEachOptions {
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. You can also specify inequality: `DateCreated<=YYYY-MM-DD` will return recordings generated at or before midnight on a given date, and `DateCreated>=YYYY-MM-DD` returns recordings generated at or after midnight on a date. */
    dateCreated?: Date;
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. You can also specify inequality: `DateCreated<=YYYY-MM-DD` will return recordings generated at or before midnight on a given date, and `DateCreated>=YYYY-MM-DD` returns recordings generated at or after midnight on a date. */
    dateCreatedBefore?: Date;
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. You can also specify inequality: `DateCreated<=YYYY-MM-DD` will return recordings generated at or before midnight on a given date, and `DateCreated>=YYYY-MM-DD` returns recordings generated at or after midnight on a date. */
    dateCreatedAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: RecordingInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface RecordingListInstanceOptions {
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. You can also specify inequality: `DateCreated<=YYYY-MM-DD` will return recordings generated at or before midnight on a given date, and `DateCreated>=YYYY-MM-DD` returns recordings generated at or after midnight on a date. */
    dateCreated?: Date;
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. You can also specify inequality: `DateCreated<=YYYY-MM-DD` will return recordings generated at or before midnight on a given date, and `DateCreated>=YYYY-MM-DD` returns recordings generated at or after midnight on a date. */
    dateCreatedBefore?: Date;
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. You can also specify inequality: `DateCreated<=YYYY-MM-DD` will return recordings generated at or before midnight on a given date, and `DateCreated>=YYYY-MM-DD` returns recordings generated at or after midnight on a date. */
    dateCreatedAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface RecordingListInstancePageOptions {
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. You can also specify inequality: `DateCreated<=YYYY-MM-DD` will return recordings generated at or before midnight on a given date, and `DateCreated>=YYYY-MM-DD` returns recordings generated at or after midnight on a date. */
    dateCreated?: Date;
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. You can also specify inequality: `DateCreated<=YYYY-MM-DD` will return recordings generated at or before midnight on a given date, and `DateCreated>=YYYY-MM-DD` returns recordings generated at or after midnight on a date. */
    dateCreatedBefore?: Date;
    /** The `date_created` value, specified as `YYYY-MM-DD`, of the resources to read. You can also specify inequality: `DateCreated<=YYYY-MM-DD` will return recordings generated at or before midnight on a given date, and `DateCreated>=YYYY-MM-DD` returns recordings generated at or after midnight on a date. */
    dateCreatedAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface RecordingContext {
    /**
     * Remove a RecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingInstance
     */
    fetch(callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Update a RecordingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingInstance
     */
    update(params: RecordingContextUpdateOptions, callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface RecordingContextSolution {
    accountSid: string;
    conferenceSid: string;
    sid: string;
}
export declare class RecordingContextImpl implements RecordingContext {
    protected _version: V2010;
    protected _solution: RecordingContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, conferenceSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    update(params: RecordingContextUpdateOptions, callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): RecordingContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface RecordingPayload extends TwilioResponsePayload {
    recordings: RecordingResource[];
}
interface RecordingResource {
    account_sid: string;
    api_version: string;
    call_sid: string;
    conference_sid: string;
    date_created: Date;
    date_updated: Date;
    start_time: Date;
    duration: string;
    sid: string;
    price: string;
    price_unit: string;
    status: RecordingStatus;
    channels: number;
    source: RecordingSource;
    error_code: number;
    encryption_details: any;
    uri: string;
}
export declare class RecordingInstance {
    protected _version: V2010;
    protected _solution: RecordingContextSolution;
    protected _context?: RecordingContext;
    constructor(_version: V2010, payload: RecordingResource, accountSid: string, conferenceSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Conference Recording resource.
     */
    accountSid: string;
    /**
     * The API version used to create the recording.
     */
    apiVersion: string;
    /**
     * The SID of the [Call](https://www.twilio.com/docs/voice/api/call-resource) the Conference Recording resource is associated with.
     */
    callSid: string;
    /**
     * The Conference SID that identifies the conference associated with the recording.
     */
    conferenceSid: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated, specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The start time of the recording in GMT and in [RFC 2822](https://www.php.net/manual/en/class.datetime.php#datetime.constants.rfc2822) format.
     */
    startTime: Date;
    /**
     * The length of the recording in seconds.
     */
    duration: string;
    /**
     * The unique string that that we created to identify the Conference Recording resource.
     */
    sid: string;
    /**
     * The one-time cost of creating the recording in the `price_unit` currency.
     */
    price: string;
    /**
     * The currency used in the `price` property. Example: `USD`.
     */
    priceUnit: string;
    status: RecordingStatus;
    /**
     * The number of channels in the final recording file.  Can be: `1`, or `2`. Separating a two leg call into two separate channels of the recording file is supported in [Dial](https://www.twilio.com/docs/voice/twiml/dial#attributes-record) and [Outbound Rest API](https://www.twilio.com/docs/voice/make-calls) record options.
     */
    channels: number;
    source: RecordingSource;
    /**
     * The error code that describes why the recording is `absent`. The error code is described in our [Error Dictionary](https://www.twilio.com/docs/api/errors). This value is null if the recording `status` is not `absent`.
     */
    errorCode: number;
    /**
     * How to decrypt the recording if it was encrypted using [Call Recording Encryption](https://www.twilio.com/docs/voice/tutorials/voice-recording-encryption) feature.
     */
    encryptionDetails: any;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Remove a RecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RecordingInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingInstance
     */
    fetch(callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Update a RecordingInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingInstance
     */
    update(params: RecordingContextUpdateOptions, callback?: (error: Error | null, item?: RecordingInstance) => any): Promise<RecordingInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        apiVersion: string;
        callSid: string;
        conferenceSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        startTime: Date;
        duration: string;
        sid: string;
        price: string;
        priceUnit: string;
        status: RecordingStatus;
        channels: number;
        source: RecordingSource;
        errorCode: number;
        encryptionDetails: any;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface RecordingSolution {
    accountSid: string;
    conferenceSid: string;
}
export interface RecordingListInstance {
    _version: V2010;
    _solution: RecordingSolution;
    _uri: string;
    (sid: string): RecordingContext;
    get(sid: string): RecordingContext;
    /**
     * Streams RecordingInstance records from the API.
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
     * @param { RecordingListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: RecordingInstance, done: (err?: Error) => void) => void): void;
    each(params: RecordingListInstanceEachOptions, callback?: (item: RecordingInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of RecordingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: RecordingPage) => any): Promise<RecordingPage>;
    /**
     * Lists RecordingInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RecordingListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: RecordingInstance[]) => any): Promise<RecordingInstance[]>;
    list(params: RecordingListInstanceOptions, callback?: (error: Error | null, items: RecordingInstance[]) => any): Promise<RecordingInstance[]>;
    /**
     * Retrieve a single page of RecordingInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RecordingListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: RecordingPage) => any): Promise<RecordingPage>;
    page(params: RecordingListInstancePageOptions, callback?: (error: Error | null, items: RecordingPage) => any): Promise<RecordingPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RecordingListInstance(version: V2010, accountSid: string, conferenceSid: string): RecordingListInstance;
export declare class RecordingPage extends Page<V2010, RecordingPayload, RecordingResource, RecordingInstance> {
    /**
     * Initialize the RecordingPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: RecordingSolution);
    /**
     * Build an instance of RecordingInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: RecordingResource): RecordingInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
