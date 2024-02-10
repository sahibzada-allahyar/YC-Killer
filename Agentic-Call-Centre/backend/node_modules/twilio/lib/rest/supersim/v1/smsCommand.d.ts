/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type SmsCommandDirection = "to_sim" | "from_sim";
export type SmsCommandStatus = "queued" | "sent" | "delivered" | "received" | "failed";
/**
 * Options to pass to create a SmsCommandInstance
 */
export interface SmsCommandListInstanceCreateOptions {
    /** The `sid` or `unique_name` of the [SIM](https://www.twilio.com/docs/iot/supersim/api/sim-resource) to send the SMS Command to. */
    sim: string;
    /** The message body of the SMS Command. */
    payload: string;
    /** The HTTP method we should use to call `callback_url`. Can be: `GET` or `POST` and the default is POST. */
    callbackMethod?: string;
    /** The URL we should call using the `callback_method` after we have sent the command. */
    callbackUrl?: string;
}
/**
 * Options to pass to each
 */
export interface SmsCommandListInstanceEachOptions {
    /** The SID or unique name of the Sim resource that SMS Command was sent to or from. */
    sim?: string;
    /** The status of the SMS Command. Can be: `queued`, `sent`, `delivered`, `received` or `failed`. See the [SMS Command Status Values](https://www.twilio.com/docs/iot/supersim/api/smscommand-resource#status-values) for a description of each. */
    status?: SmsCommandStatus;
    /** The direction of the SMS Command. Can be `to_sim` or `from_sim`. The value of `to_sim` is synonymous with the term `mobile terminated`, and `from_sim` is synonymous with the term `mobile originated`. */
    direction?: SmsCommandDirection;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SmsCommandInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SmsCommandListInstanceOptions {
    /** The SID or unique name of the Sim resource that SMS Command was sent to or from. */
    sim?: string;
    /** The status of the SMS Command. Can be: `queued`, `sent`, `delivered`, `received` or `failed`. See the [SMS Command Status Values](https://www.twilio.com/docs/iot/supersim/api/smscommand-resource#status-values) for a description of each. */
    status?: SmsCommandStatus;
    /** The direction of the SMS Command. Can be `to_sim` or `from_sim`. The value of `to_sim` is synonymous with the term `mobile terminated`, and `from_sim` is synonymous with the term `mobile originated`. */
    direction?: SmsCommandDirection;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SmsCommandListInstancePageOptions {
    /** The SID or unique name of the Sim resource that SMS Command was sent to or from. */
    sim?: string;
    /** The status of the SMS Command. Can be: `queued`, `sent`, `delivered`, `received` or `failed`. See the [SMS Command Status Values](https://www.twilio.com/docs/iot/supersim/api/smscommand-resource#status-values) for a description of each. */
    status?: SmsCommandStatus;
    /** The direction of the SMS Command. Can be `to_sim` or `from_sim`. The value of `to_sim` is synonymous with the term `mobile terminated`, and `from_sim` is synonymous with the term `mobile originated`. */
    direction?: SmsCommandDirection;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SmsCommandContext {
    /**
     * Fetch a SmsCommandInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SmsCommandInstance
     */
    fetch(callback?: (error: Error | null, item?: SmsCommandInstance) => any): Promise<SmsCommandInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SmsCommandContextSolution {
    sid: string;
}
export declare class SmsCommandContextImpl implements SmsCommandContext {
    protected _version: V1;
    protected _solution: SmsCommandContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: SmsCommandInstance) => any): Promise<SmsCommandInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SmsCommandContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SmsCommandPayload extends TwilioResponsePayload {
    sms_commands: SmsCommandResource[];
}
interface SmsCommandResource {
    sid: string;
    account_sid: string;
    sim_sid: string;
    payload: string;
    status: SmsCommandStatus;
    direction: SmsCommandDirection;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class SmsCommandInstance {
    protected _version: V1;
    protected _solution: SmsCommandContextSolution;
    protected _context?: SmsCommandContext;
    constructor(_version: V1, payload: SmsCommandResource, sid?: string);
    /**
     * The unique string that we created to identify the SMS Command resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the SMS Command resource.
     */
    accountSid: string;
    /**
     * The SID of the [SIM](https://www.twilio.com/docs/iot/supersim/api/sim-resource) that this SMS Command was sent to or from.
     */
    simSid: string;
    /**
     * The message body of the SMS Command sent to or from the SIM. For text mode messages, this can be up to 160 characters.
     */
    payload: string;
    status: SmsCommandStatus;
    direction: SmsCommandDirection;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the SMS Command resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a SmsCommandInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SmsCommandInstance
     */
    fetch(callback?: (error: Error | null, item?: SmsCommandInstance) => any): Promise<SmsCommandInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        simSid: string;
        payload: string;
        status: SmsCommandStatus;
        direction: SmsCommandDirection;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SmsCommandSolution {
}
export interface SmsCommandListInstance {
    _version: V1;
    _solution: SmsCommandSolution;
    _uri: string;
    (sid: string): SmsCommandContext;
    get(sid: string): SmsCommandContext;
    /**
     * Create a SmsCommandInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SmsCommandInstance
     */
    create(params: SmsCommandListInstanceCreateOptions, callback?: (error: Error | null, item?: SmsCommandInstance) => any): Promise<SmsCommandInstance>;
    /**
     * Streams SmsCommandInstance records from the API.
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
     * @param { SmsCommandListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SmsCommandInstance, done: (err?: Error) => void) => void): void;
    each(params: SmsCommandListInstanceEachOptions, callback?: (item: SmsCommandInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SmsCommandInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SmsCommandPage) => any): Promise<SmsCommandPage>;
    /**
     * Lists SmsCommandInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SmsCommandListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SmsCommandInstance[]) => any): Promise<SmsCommandInstance[]>;
    list(params: SmsCommandListInstanceOptions, callback?: (error: Error | null, items: SmsCommandInstance[]) => any): Promise<SmsCommandInstance[]>;
    /**
     * Retrieve a single page of SmsCommandInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SmsCommandListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SmsCommandPage) => any): Promise<SmsCommandPage>;
    page(params: SmsCommandListInstancePageOptions, callback?: (error: Error | null, items: SmsCommandPage) => any): Promise<SmsCommandPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SmsCommandListInstance(version: V1): SmsCommandListInstance;
export declare class SmsCommandPage extends Page<V1, SmsCommandPayload, SmsCommandResource, SmsCommandInstance> {
    /**
     * Initialize the SmsCommandPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SmsCommandSolution);
    /**
     * Build an instance of SmsCommandInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SmsCommandResource): SmsCommandInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
