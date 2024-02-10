/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { ApprovalFetchListInstance } from "./content/approvalFetch";
/**
 * Options to pass to each
 */
export interface ContentListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ContentInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ContentListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ContentListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ContentContext {
    approvalFetch: ApprovalFetchListInstance;
    /**
     * Remove a ContentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ContentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ContentInstance
     */
    fetch(callback?: (error: Error | null, item?: ContentInstance) => any): Promise<ContentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ContentContextSolution {
    sid: string;
}
export declare class ContentContextImpl implements ContentContext {
    protected _version: V1;
    protected _solution: ContentContextSolution;
    protected _uri: string;
    protected _approvalFetch?: ApprovalFetchListInstance;
    constructor(_version: V1, sid: string);
    get approvalFetch(): ApprovalFetchListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ContentInstance) => any): Promise<ContentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ContentContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ContentPayload extends TwilioResponsePayload {
    contents: ContentResource[];
}
interface ContentResource {
    date_created: Date;
    date_updated: Date;
    sid: string;
    account_sid: string;
    friendly_name: string;
    language: string;
    variables: any;
    types: any;
    url: string;
    links: Record<string, string>;
}
export declare class ContentInstance {
    protected _version: V1;
    protected _solution: ContentContextSolution;
    protected _context?: ContentContext;
    constructor(_version: V1, payload: ContentResource, sid?: string);
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The unique string that that we created to identify the Content resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/usage/api/account) that created Content resource.
     */
    accountSid: string;
    /**
     * A string name used to describe the Content resource. Not visible to the end recipient.
     */
    friendlyName: string;
    /**
     * Two-letter (ISO 639-1) language code (e.g., en) identifying the language the Content resource is in.
     */
    language: string;
    /**
     * Defines the default placeholder values for variables included in the Content resource. e.g. {\"1\": \"Customer_Name\"}.
     */
    variables: any;
    /**
     * The [Content types](https://www.twilio.com/docs/content/content-types-overview) (e.g. twilio/text) for this Content resource.
     */
    types: any;
    /**
     * The URL of the resource, relative to `https://content.twilio.com`.
     */
    url: string;
    /**
     * A list of links related to the Content resource, such as approval_fetch and approval_create
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a ContentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ContentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ContentInstance
     */
    fetch(callback?: (error: Error | null, item?: ContentInstance) => any): Promise<ContentInstance>;
    /**
     * Access the approvalFetch.
     */
    approvalFetch(): ApprovalFetchListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        dateCreated: Date;
        dateUpdated: Date;
        sid: string;
        accountSid: string;
        friendlyName: string;
        language: string;
        variables: any;
        types: any;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ContentSolution {
}
export interface ContentListInstance {
    _version: V1;
    _solution: ContentSolution;
    _uri: string;
    (sid: string): ContentContext;
    get(sid: string): ContentContext;
    /**
     * Streams ContentInstance records from the API.
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
     * @param { ContentListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ContentInstance, done: (err?: Error) => void) => void): void;
    each(params: ContentListInstanceEachOptions, callback?: (item: ContentInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ContentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ContentPage) => any): Promise<ContentPage>;
    /**
     * Lists ContentInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ContentListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ContentInstance[]) => any): Promise<ContentInstance[]>;
    list(params: ContentListInstanceOptions, callback?: (error: Error | null, items: ContentInstance[]) => any): Promise<ContentInstance[]>;
    /**
     * Retrieve a single page of ContentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ContentListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ContentPage) => any): Promise<ContentPage>;
    page(params: ContentListInstancePageOptions, callback?: (error: Error | null, items: ContentPage) => any): Promise<ContentPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ContentListInstance(version: V1): ContentListInstance;
export declare class ContentPage extends Page<V1, ContentPayload, ContentResource, ContentInstance> {
    /**
     * Initialize the ContentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ContentSolution);
    /**
     * Build an instance of ContentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ContentResource): ContentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
