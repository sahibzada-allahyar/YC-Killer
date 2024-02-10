/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../../base/Page";
import Response from "../../../../../../http/response";
import V2010 from "../../../../V2010";
/**
 * Options to pass to each
 */
export interface PayloadListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: PayloadInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface PayloadListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface PayloadListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface PayloadContext {
    /**
     * Remove a PayloadInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a PayloadInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PayloadInstance
     */
    fetch(callback?: (error: Error | null, item?: PayloadInstance) => any): Promise<PayloadInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PayloadContextSolution {
    accountSid: string;
    referenceSid: string;
    addOnResultSid: string;
    sid: string;
}
export declare class PayloadContextImpl implements PayloadContext {
    protected _version: V2010;
    protected _solution: PayloadContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, referenceSid: string, addOnResultSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: PayloadInstance) => any): Promise<PayloadInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PayloadContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PayloadPayload extends TwilioResponsePayload {
    payloads: PayloadResource[];
}
interface PayloadResource {
    sid: string;
    add_on_result_sid: string;
    account_sid: string;
    label: string;
    add_on_sid: string;
    add_on_configuration_sid: string;
    content_type: string;
    date_created: Date;
    date_updated: Date;
    reference_sid: string;
    subresource_uris: Record<string, string>;
}
export declare class PayloadInstance {
    protected _version: V2010;
    protected _solution: PayloadContextSolution;
    protected _context?: PayloadContext;
    constructor(_version: V2010, payload: PayloadResource, accountSid: string, referenceSid: string, addOnResultSid: string, sid?: string);
    /**
     * The unique string that that we created to identify the Recording AddOnResult Payload resource.
     */
    sid: string;
    /**
     * The SID of the AddOnResult to which the payload belongs.
     */
    addOnResultSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Recording AddOnResult Payload resource.
     */
    accountSid: string;
    /**
     * The string provided by the vendor that describes the payload.
     */
    label: string;
    /**
     * The SID of the Add-on to which the result belongs.
     */
    addOnSid: string;
    /**
     * The SID of the Add-on configuration.
     */
    addOnConfigurationSid: string;
    /**
     * The MIME type of the payload.
     */
    contentType: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The SID of the recording to which the AddOnResult resource that contains the payload belongs.
     */
    referenceSid: string;
    /**
     * A list of related resources identified by their relative URIs.
     */
    subresourceUris: Record<string, string>;
    private get _proxy();
    /**
     * Remove a PayloadInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a PayloadInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PayloadInstance
     */
    fetch(callback?: (error: Error | null, item?: PayloadInstance) => any): Promise<PayloadInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        addOnResultSid: string;
        accountSid: string;
        label: string;
        addOnSid: string;
        addOnConfigurationSid: string;
        contentType: string;
        dateCreated: Date;
        dateUpdated: Date;
        referenceSid: string;
        subresourceUris: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PayloadSolution {
    accountSid: string;
    referenceSid: string;
    addOnResultSid: string;
}
export interface PayloadListInstance {
    _version: V2010;
    _solution: PayloadSolution;
    _uri: string;
    (sid: string): PayloadContext;
    get(sid: string): PayloadContext;
    /**
     * Streams PayloadInstance records from the API.
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
     * @param { PayloadListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: PayloadInstance, done: (err?: Error) => void) => void): void;
    each(params: PayloadListInstanceEachOptions, callback?: (item: PayloadInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of PayloadInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: PayloadPage) => any): Promise<PayloadPage>;
    /**
     * Lists PayloadInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PayloadListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: PayloadInstance[]) => any): Promise<PayloadInstance[]>;
    list(params: PayloadListInstanceOptions, callback?: (error: Error | null, items: PayloadInstance[]) => any): Promise<PayloadInstance[]>;
    /**
     * Retrieve a single page of PayloadInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PayloadListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: PayloadPage) => any): Promise<PayloadPage>;
    page(params: PayloadListInstancePageOptions, callback?: (error: Error | null, items: PayloadPage) => any): Promise<PayloadPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PayloadListInstance(version: V2010, accountSid: string, referenceSid: string, addOnResultSid: string): PayloadListInstance;
export declare class PayloadPage extends Page<V2010, PayloadPayload, PayloadResource, PayloadInstance> {
    /**
     * Initialize the PayloadPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: PayloadSolution);
    /**
     * Build an instance of PayloadInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: PayloadResource): PayloadInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
