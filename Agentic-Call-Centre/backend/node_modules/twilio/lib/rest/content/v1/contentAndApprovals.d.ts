/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to each
 */
export interface ContentAndApprovalsListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ContentAndApprovalsInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ContentAndApprovalsListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ContentAndApprovalsListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ContentAndApprovalsSolution {
}
export interface ContentAndApprovalsListInstance {
    _version: V1;
    _solution: ContentAndApprovalsSolution;
    _uri: string;
    /**
     * Streams ContentAndApprovalsInstance records from the API.
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
     * @param { ContentAndApprovalsListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ContentAndApprovalsInstance, done: (err?: Error) => void) => void): void;
    each(params: ContentAndApprovalsListInstanceEachOptions, callback?: (item: ContentAndApprovalsInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ContentAndApprovalsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ContentAndApprovalsPage) => any): Promise<ContentAndApprovalsPage>;
    /**
     * Lists ContentAndApprovalsInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ContentAndApprovalsListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ContentAndApprovalsInstance[]) => any): Promise<ContentAndApprovalsInstance[]>;
    list(params: ContentAndApprovalsListInstanceOptions, callback?: (error: Error | null, items: ContentAndApprovalsInstance[]) => any): Promise<ContentAndApprovalsInstance[]>;
    /**
     * Retrieve a single page of ContentAndApprovalsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ContentAndApprovalsListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ContentAndApprovalsPage) => any): Promise<ContentAndApprovalsPage>;
    page(params: ContentAndApprovalsListInstancePageOptions, callback?: (error: Error | null, items: ContentAndApprovalsPage) => any): Promise<ContentAndApprovalsPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ContentAndApprovalsListInstance(version: V1): ContentAndApprovalsListInstance;
interface ContentAndApprovalsPayload extends TwilioResponsePayload {
    contents: ContentAndApprovalsResource[];
}
interface ContentAndApprovalsResource {
    date_created: Date;
    date_updated: Date;
    sid: string;
    account_sid: string;
    friendly_name: string;
    language: string;
    variables: any;
    types: any;
    approval_requests: any;
}
export declare class ContentAndApprovalsInstance {
    protected _version: V1;
    constructor(_version: V1, payload: ContentAndApprovalsResource);
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
     * The submitted information and approval request status of the Content resource.
     */
    approvalRequests: any;
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
        approvalRequests: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class ContentAndApprovalsPage extends Page<V1, ContentAndApprovalsPayload, ContentAndApprovalsResource, ContentAndApprovalsInstance> {
    /**
     * Initialize the ContentAndApprovalsPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ContentAndApprovalsSolution);
    /**
     * Build an instance of ContentAndApprovalsInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ContentAndApprovalsResource): ContentAndApprovalsInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
