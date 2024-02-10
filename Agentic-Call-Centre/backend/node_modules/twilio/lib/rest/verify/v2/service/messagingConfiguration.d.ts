/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
/**
 * Options to pass to update a MessagingConfigurationInstance
 */
export interface MessagingConfigurationContextUpdateOptions {
    /** The SID of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) to be used to send SMS to the country of this configuration. */
    messagingServiceSid: string;
}
/**
 * Options to pass to create a MessagingConfigurationInstance
 */
export interface MessagingConfigurationListInstanceCreateOptions {
    /** The [ISO-3166-1](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) country code of the country this configuration will be applied to. If this is a global configuration, Country will take the value `all`. */
    country: string;
    /** The SID of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) to be used to send SMS to the country of this configuration. */
    messagingServiceSid: string;
}
/**
 * Options to pass to each
 */
export interface MessagingConfigurationListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: MessagingConfigurationInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface MessagingConfigurationListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface MessagingConfigurationListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface MessagingConfigurationContext {
    /**
     * Remove a MessagingConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a MessagingConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessagingConfigurationInstance
     */
    fetch(callback?: (error: Error | null, item?: MessagingConfigurationInstance) => any): Promise<MessagingConfigurationInstance>;
    /**
     * Update a MessagingConfigurationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessagingConfigurationInstance
     */
    update(params: MessagingConfigurationContextUpdateOptions, callback?: (error: Error | null, item?: MessagingConfigurationInstance) => any): Promise<MessagingConfigurationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface MessagingConfigurationContextSolution {
    serviceSid: string;
    country: string;
}
export declare class MessagingConfigurationContextImpl implements MessagingConfigurationContext {
    protected _version: V2;
    protected _solution: MessagingConfigurationContextSolution;
    protected _uri: string;
    constructor(_version: V2, serviceSid: string, country: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: MessagingConfigurationInstance) => any): Promise<MessagingConfigurationInstance>;
    update(params: MessagingConfigurationContextUpdateOptions, callback?: (error: Error | null, item?: MessagingConfigurationInstance) => any): Promise<MessagingConfigurationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): MessagingConfigurationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface MessagingConfigurationPayload extends TwilioResponsePayload {
    messaging_configurations: MessagingConfigurationResource[];
}
interface MessagingConfigurationResource {
    account_sid: string;
    service_sid: string;
    country: string;
    messaging_service_sid: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class MessagingConfigurationInstance {
    protected _version: V2;
    protected _solution: MessagingConfigurationContextSolution;
    protected _context?: MessagingConfigurationContext;
    constructor(_version: V2, payload: MessagingConfigurationResource, serviceSid: string, country?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Service resource.
     */
    accountSid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/verify/api/service) that the resource is associated with.
     */
    serviceSid: string;
    /**
     * The [ISO-3166-1](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) country code of the country this configuration will be applied to. If this is a global configuration, Country will take the value `all`.
     */
    country: string;
    /**
     * The SID of the [Messaging Service](https://www.twilio.com/docs/messaging/api/service-resource) to be used to send SMS to the country of this configuration.
     */
    messagingServiceSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The URL of this resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a MessagingConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a MessagingConfigurationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessagingConfigurationInstance
     */
    fetch(callback?: (error: Error | null, item?: MessagingConfigurationInstance) => any): Promise<MessagingConfigurationInstance>;
    /**
     * Update a MessagingConfigurationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessagingConfigurationInstance
     */
    update(params: MessagingConfigurationContextUpdateOptions, callback?: (error: Error | null, item?: MessagingConfigurationInstance) => any): Promise<MessagingConfigurationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        serviceSid: string;
        country: string;
        messagingServiceSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface MessagingConfigurationSolution {
    serviceSid: string;
}
export interface MessagingConfigurationListInstance {
    _version: V2;
    _solution: MessagingConfigurationSolution;
    _uri: string;
    (country: string): MessagingConfigurationContext;
    get(country: string): MessagingConfigurationContext;
    /**
     * Create a MessagingConfigurationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessagingConfigurationInstance
     */
    create(params: MessagingConfigurationListInstanceCreateOptions, callback?: (error: Error | null, item?: MessagingConfigurationInstance) => any): Promise<MessagingConfigurationInstance>;
    /**
     * Streams MessagingConfigurationInstance records from the API.
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
     * @param { MessagingConfigurationListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: MessagingConfigurationInstance, done: (err?: Error) => void) => void): void;
    each(params: MessagingConfigurationListInstanceEachOptions, callback?: (item: MessagingConfigurationInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of MessagingConfigurationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: MessagingConfigurationPage) => any): Promise<MessagingConfigurationPage>;
    /**
     * Lists MessagingConfigurationInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MessagingConfigurationListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: MessagingConfigurationInstance[]) => any): Promise<MessagingConfigurationInstance[]>;
    list(params: MessagingConfigurationListInstanceOptions, callback?: (error: Error | null, items: MessagingConfigurationInstance[]) => any): Promise<MessagingConfigurationInstance[]>;
    /**
     * Retrieve a single page of MessagingConfigurationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MessagingConfigurationListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: MessagingConfigurationPage) => any): Promise<MessagingConfigurationPage>;
    page(params: MessagingConfigurationListInstancePageOptions, callback?: (error: Error | null, items: MessagingConfigurationPage) => any): Promise<MessagingConfigurationPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function MessagingConfigurationListInstance(version: V2, serviceSid: string): MessagingConfigurationListInstance;
export declare class MessagingConfigurationPage extends Page<V2, MessagingConfigurationPayload, MessagingConfigurationResource, MessagingConfigurationInstance> {
    /**
     * Initialize the MessagingConfigurationPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: MessagingConfigurationSolution);
    /**
     * Build an instance of MessagingConfigurationInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: MessagingConfigurationResource): MessagingConfigurationInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
