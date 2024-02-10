/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
/**
 * Options to pass to update a OutgoingCallerIdInstance
 */
export interface OutgoingCallerIdContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
}
/**
 * Options to pass to each
 */
export interface OutgoingCallerIdListInstanceEachOptions {
    /** The phone number of the OutgoingCallerId resources to read. */
    phoneNumber?: string;
    /** The string that identifies the OutgoingCallerId resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: OutgoingCallerIdInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface OutgoingCallerIdListInstanceOptions {
    /** The phone number of the OutgoingCallerId resources to read. */
    phoneNumber?: string;
    /** The string that identifies the OutgoingCallerId resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface OutgoingCallerIdListInstancePageOptions {
    /** The phone number of the OutgoingCallerId resources to read. */
    phoneNumber?: string;
    /** The string that identifies the OutgoingCallerId resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface OutgoingCallerIdContext {
    /**
     * Remove a OutgoingCallerIdInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a OutgoingCallerIdInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OutgoingCallerIdInstance
     */
    fetch(callback?: (error: Error | null, item?: OutgoingCallerIdInstance) => any): Promise<OutgoingCallerIdInstance>;
    /**
     * Update a OutgoingCallerIdInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OutgoingCallerIdInstance
     */
    update(callback?: (error: Error | null, item?: OutgoingCallerIdInstance) => any): Promise<OutgoingCallerIdInstance>;
    /**
     * Update a OutgoingCallerIdInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OutgoingCallerIdInstance
     */
    update(params: OutgoingCallerIdContextUpdateOptions, callback?: (error: Error | null, item?: OutgoingCallerIdInstance) => any): Promise<OutgoingCallerIdInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface OutgoingCallerIdContextSolution {
    accountSid: string;
    sid: string;
}
export declare class OutgoingCallerIdContextImpl implements OutgoingCallerIdContext {
    protected _version: V2010;
    protected _solution: OutgoingCallerIdContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: OutgoingCallerIdInstance) => any): Promise<OutgoingCallerIdInstance>;
    update(params?: OutgoingCallerIdContextUpdateOptions | ((error: Error | null, item?: OutgoingCallerIdInstance) => any), callback?: (error: Error | null, item?: OutgoingCallerIdInstance) => any): Promise<OutgoingCallerIdInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): OutgoingCallerIdContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface OutgoingCallerIdPayload extends TwilioResponsePayload {
    outgoing_caller_ids: OutgoingCallerIdResource[];
}
interface OutgoingCallerIdResource {
    sid: string;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    account_sid: string;
    phone_number: string;
    uri: string;
}
export declare class OutgoingCallerIdInstance {
    protected _version: V2010;
    protected _solution: OutgoingCallerIdContextSolution;
    protected _context?: OutgoingCallerIdContext;
    constructor(_version: V2010, payload: OutgoingCallerIdResource, accountSid: string, sid?: string);
    /**
     * The unique string that that we created to identify the OutgoingCallerId resource.
     */
    sid: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the OutgoingCallerId resource.
     */
    accountSid: string;
    /**
     * The phone number in [E.164](https://www.twilio.com/docs/glossary/what-e164) format, which consists of a + followed by the country code and subscriber number.
     */
    phoneNumber: string;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Remove a OutgoingCallerIdInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a OutgoingCallerIdInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OutgoingCallerIdInstance
     */
    fetch(callback?: (error: Error | null, item?: OutgoingCallerIdInstance) => any): Promise<OutgoingCallerIdInstance>;
    /**
     * Update a OutgoingCallerIdInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OutgoingCallerIdInstance
     */
    update(callback?: (error: Error | null, item?: OutgoingCallerIdInstance) => any): Promise<OutgoingCallerIdInstance>;
    /**
     * Update a OutgoingCallerIdInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed OutgoingCallerIdInstance
     */
    update(params: OutgoingCallerIdContextUpdateOptions, callback?: (error: Error | null, item?: OutgoingCallerIdInstance) => any): Promise<OutgoingCallerIdInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        accountSid: string;
        phoneNumber: string;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface OutgoingCallerIdSolution {
    accountSid: string;
}
export interface OutgoingCallerIdListInstance {
    _version: V2010;
    _solution: OutgoingCallerIdSolution;
    _uri: string;
    (sid: string): OutgoingCallerIdContext;
    get(sid: string): OutgoingCallerIdContext;
    /**
     * Streams OutgoingCallerIdInstance records from the API.
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
     * @param { OutgoingCallerIdListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: OutgoingCallerIdInstance, done: (err?: Error) => void) => void): void;
    each(params: OutgoingCallerIdListInstanceEachOptions, callback?: (item: OutgoingCallerIdInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of OutgoingCallerIdInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: OutgoingCallerIdPage) => any): Promise<OutgoingCallerIdPage>;
    /**
     * Lists OutgoingCallerIdInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { OutgoingCallerIdListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: OutgoingCallerIdInstance[]) => any): Promise<OutgoingCallerIdInstance[]>;
    list(params: OutgoingCallerIdListInstanceOptions, callback?: (error: Error | null, items: OutgoingCallerIdInstance[]) => any): Promise<OutgoingCallerIdInstance[]>;
    /**
     * Retrieve a single page of OutgoingCallerIdInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { OutgoingCallerIdListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: OutgoingCallerIdPage) => any): Promise<OutgoingCallerIdPage>;
    page(params: OutgoingCallerIdListInstancePageOptions, callback?: (error: Error | null, items: OutgoingCallerIdPage) => any): Promise<OutgoingCallerIdPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function OutgoingCallerIdListInstance(version: V2010, accountSid: string): OutgoingCallerIdListInstance;
export declare class OutgoingCallerIdPage extends Page<V2010, OutgoingCallerIdPayload, OutgoingCallerIdResource, OutgoingCallerIdInstance> {
    /**
     * Initialize the OutgoingCallerIdPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: OutgoingCallerIdSolution);
    /**
     * Build an instance of OutgoingCallerIdInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: OutgoingCallerIdResource): OutgoingCallerIdInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
