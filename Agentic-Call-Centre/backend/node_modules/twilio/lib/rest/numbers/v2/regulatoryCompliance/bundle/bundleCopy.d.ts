/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2 from "../../../V2";
export type BundleCopyStatus = "draft" | "pending-review" | "in-review" | "twilio-rejected" | "twilio-approved" | "provisionally-approved";
/**
 * Options to pass to create a BundleCopyInstance
 */
export interface BundleCopyListInstanceCreateOptions {
    /** The string that you assigned to describe the copied bundle. */
    friendlyName?: string;
}
/**
 * Options to pass to each
 */
export interface BundleCopyListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: BundleCopyInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface BundleCopyListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface BundleCopyListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface BundleCopySolution {
    bundleSid: string;
}
export interface BundleCopyListInstance {
    _version: V2;
    _solution: BundleCopySolution;
    _uri: string;
    /**
     * Create a BundleCopyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BundleCopyInstance
     */
    create(callback?: (error: Error | null, item?: BundleCopyInstance) => any): Promise<BundleCopyInstance>;
    /**
     * Create a BundleCopyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BundleCopyInstance
     */
    create(params: BundleCopyListInstanceCreateOptions, callback?: (error: Error | null, item?: BundleCopyInstance) => any): Promise<BundleCopyInstance>;
    /**
     * Streams BundleCopyInstance records from the API.
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
     * @param { BundleCopyListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: BundleCopyInstance, done: (err?: Error) => void) => void): void;
    each(params: BundleCopyListInstanceEachOptions, callback?: (item: BundleCopyInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of BundleCopyInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: BundleCopyPage) => any): Promise<BundleCopyPage>;
    /**
     * Lists BundleCopyInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BundleCopyListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: BundleCopyInstance[]) => any): Promise<BundleCopyInstance[]>;
    list(params: BundleCopyListInstanceOptions, callback?: (error: Error | null, items: BundleCopyInstance[]) => any): Promise<BundleCopyInstance[]>;
    /**
     * Retrieve a single page of BundleCopyInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BundleCopyListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: BundleCopyPage) => any): Promise<BundleCopyPage>;
    page(params: BundleCopyListInstancePageOptions, callback?: (error: Error | null, items: BundleCopyPage) => any): Promise<BundleCopyPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BundleCopyListInstance(version: V2, bundleSid: string): BundleCopyListInstance;
interface BundleCopyPayload extends TwilioResponsePayload {
    results: BundleCopyResource[];
}
interface BundleCopyResource {
    sid: string;
    account_sid: string;
    regulation_sid: string;
    friendly_name: string;
    status: BundleCopyStatus;
    valid_until: Date;
    email: string;
    status_callback: string;
    date_created: Date;
    date_updated: Date;
}
export declare class BundleCopyInstance {
    protected _version: V2;
    constructor(_version: V2, payload: BundleCopyResource, bundleSid: string);
    /**
     * The unique string that we created to identify the Bundle resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Bundle resource.
     */
    accountSid: string;
    /**
     * The unique string of a regulation that is associated to the Bundle resource.
     */
    regulationSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    status: BundleCopyStatus;
    /**
     * The date and time in GMT in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format when the resource will be valid until.
     */
    validUntil: Date;
    /**
     * The email address that will receive updates when the Bundle resource changes status.
     */
    email: string;
    /**
     * The URL we call to inform your application of status changes.
     */
    statusCallback: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        regulationSid: string;
        friendlyName: string;
        status: BundleCopyStatus;
        validUntil: Date;
        email: string;
        statusCallback: string;
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class BundleCopyPage extends Page<V2, BundleCopyPayload, BundleCopyResource, BundleCopyInstance> {
    /**
     * Initialize the BundleCopyPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: BundleCopySolution);
    /**
     * Build an instance of BundleCopyInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: BundleCopyResource): BundleCopyInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
