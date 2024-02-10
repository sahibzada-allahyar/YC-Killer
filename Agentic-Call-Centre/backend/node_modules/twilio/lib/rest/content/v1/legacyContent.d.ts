/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to each
 */
export interface LegacyContentListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: LegacyContentInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface LegacyContentListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface LegacyContentListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface LegacyContentSolution {
}
export interface LegacyContentListInstance {
    _version: V1;
    _solution: LegacyContentSolution;
    _uri: string;
    /**
     * Streams LegacyContentInstance records from the API.
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
     * @param { LegacyContentListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: LegacyContentInstance, done: (err?: Error) => void) => void): void;
    each(params: LegacyContentListInstanceEachOptions, callback?: (item: LegacyContentInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of LegacyContentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: LegacyContentPage) => any): Promise<LegacyContentPage>;
    /**
     * Lists LegacyContentInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { LegacyContentListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: LegacyContentInstance[]) => any): Promise<LegacyContentInstance[]>;
    list(params: LegacyContentListInstanceOptions, callback?: (error: Error | null, items: LegacyContentInstance[]) => any): Promise<LegacyContentInstance[]>;
    /**
     * Retrieve a single page of LegacyContentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { LegacyContentListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: LegacyContentPage) => any): Promise<LegacyContentPage>;
    page(params: LegacyContentListInstancePageOptions, callback?: (error: Error | null, items: LegacyContentPage) => any): Promise<LegacyContentPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function LegacyContentListInstance(version: V1): LegacyContentListInstance;
interface LegacyContentPayload extends TwilioResponsePayload {
    contents: LegacyContentResource[];
}
interface LegacyContentResource {
    date_created: Date;
    date_updated: Date;
    sid: string;
    account_sid: string;
    friendly_name: string;
    language: string;
    variables: any;
    types: any;
    legacy_template_name: string;
    legacy_body: string;
    url: string;
}
export declare class LegacyContentInstance {
    protected _version: V1;
    constructor(_version: V1, payload: LegacyContentResource);
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
     * The string name of the legacy content template associated with this Content resource, unique across all template names for its account.  Only lowercase letters, numbers and underscores are allowed
     */
    legacyTemplateName: string;
    /**
     * The string body field of the legacy content template associated with this Content resource
     */
    legacyBody: string;
    /**
     * The URL of the resource, relative to `https://content.twilio.com`.
     */
    url: string;
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
        legacyTemplateName: string;
        legacyBody: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class LegacyContentPage extends Page<V1, LegacyContentPayload, LegacyContentResource, LegacyContentInstance> {
    /**
     * Initialize the LegacyContentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: LegacyContentSolution);
    /**
     * Build an instance of LegacyContentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: LegacyContentResource): LegacyContentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
