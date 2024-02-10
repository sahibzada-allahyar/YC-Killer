/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type FleetDataMetering = "payg";
/**
 * Options to pass to update a FleetInstance
 */
export interface FleetContextUpdateOptions {
    /** An application-defined string that uniquely identifies the resource. It can be used in place of the resource\\\'s `sid` in the URL to address the resource. */
    uniqueName?: string;
    /** The SID or unique name of the Network Access Profile that will control which cellular networks the Fleet\\\'s SIMs can connect to. */
    networkAccessProfile?: string;
    /** The URL that will receive a webhook when a Super SIM in the Fleet is used to send an IP Command from your device to a special IP address. Your server should respond with an HTTP status code in the 200 range; any response body will be ignored. */
    ipCommandsUrl?: string;
    /** A string representing the HTTP method to use when making a request to `ip_commands_url`. Can be one of `POST` or `GET`. Defaults to `POST`. */
    ipCommandsMethod?: string;
    /** The URL that will receive a webhook when a Super SIM in the Fleet is used to send an SMS from your device to the SMS Commands number. Your server should respond with an HTTP status code in the 200 range; any response body will be ignored. */
    smsCommandsUrl?: string;
    /** A string representing the HTTP method to use when making a request to `sms_commands_url`. Can be one of `POST` or `GET`. Defaults to `POST`. */
    smsCommandsMethod?: string;
    /** The total data usage (download and upload combined) in Megabytes that each Super SIM assigned to the Fleet can consume during a billing period (normally one month). Value must be between 1MB (1) and 2TB (2,000,000). Defaults to 1GB (1,000). */
    dataLimit?: number;
}
/**
 * Options to pass to create a FleetInstance
 */
export interface FleetListInstanceCreateOptions {
    /** The SID or unique name of the Network Access Profile that will control which cellular networks the Fleet\\\'s SIMs can connect to. */
    networkAccessProfile: string;
    /** An application-defined string that uniquely identifies the resource. It can be used in place of the resource\\\'s `sid` in the URL to address the resource. */
    uniqueName?: string;
    /** Defines whether SIMs in the Fleet are capable of using 2G/3G/4G/LTE/CAT-M data connectivity. Defaults to `true`. */
    dataEnabled?: boolean;
    /** The total data usage (download and upload combined) in Megabytes that each Super SIM assigned to the Fleet can consume during a billing period (normally one month). Value must be between 1MB (1) and 2TB (2,000,000). Defaults to 1GB (1,000). */
    dataLimit?: number;
    /** The URL that will receive a webhook when a Super SIM in the Fleet is used to send an IP Command from your device to a special IP address. Your server should respond with an HTTP status code in the 200 range; any response body will be ignored. */
    ipCommandsUrl?: string;
    /** A string representing the HTTP method to use when making a request to `ip_commands_url`. Can be one of `POST` or `GET`. Defaults to `POST`. */
    ipCommandsMethod?: string;
    /** Defines whether SIMs in the Fleet are capable of sending and receiving machine-to-machine SMS via Commands. Defaults to `true`. */
    smsCommandsEnabled?: boolean;
    /** The URL that will receive a webhook when a Super SIM in the Fleet is used to send an SMS from your device to the SMS Commands number. Your server should respond with an HTTP status code in the 200 range; any response body will be ignored. */
    smsCommandsUrl?: string;
    /** A string representing the HTTP method to use when making a request to `sms_commands_url`. Can be one of `POST` or `GET`. Defaults to `POST`. */
    smsCommandsMethod?: string;
}
/**
 * Options to pass to each
 */
export interface FleetListInstanceEachOptions {
    /** The SID or unique name of the Network Access Profile that controls which cellular networks the Fleet\'s SIMs can connect to. */
    networkAccessProfile?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: FleetInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface FleetListInstanceOptions {
    /** The SID or unique name of the Network Access Profile that controls which cellular networks the Fleet\'s SIMs can connect to. */
    networkAccessProfile?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface FleetListInstancePageOptions {
    /** The SID or unique name of the Network Access Profile that controls which cellular networks the Fleet\'s SIMs can connect to. */
    networkAccessProfile?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface FleetContext {
    /**
     * Fetch a FleetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FleetInstance
     */
    fetch(callback?: (error: Error | null, item?: FleetInstance) => any): Promise<FleetInstance>;
    /**
     * Update a FleetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FleetInstance
     */
    update(callback?: (error: Error | null, item?: FleetInstance) => any): Promise<FleetInstance>;
    /**
     * Update a FleetInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FleetInstance
     */
    update(params: FleetContextUpdateOptions, callback?: (error: Error | null, item?: FleetInstance) => any): Promise<FleetInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface FleetContextSolution {
    sid: string;
}
export declare class FleetContextImpl implements FleetContext {
    protected _version: V1;
    protected _solution: FleetContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: FleetInstance) => any): Promise<FleetInstance>;
    update(params?: FleetContextUpdateOptions | ((error: Error | null, item?: FleetInstance) => any), callback?: (error: Error | null, item?: FleetInstance) => any): Promise<FleetInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): FleetContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface FleetPayload extends TwilioResponsePayload {
    fleets: FleetResource[];
}
interface FleetResource {
    account_sid: string;
    sid: string;
    unique_name: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    data_enabled: boolean;
    data_limit: number;
    data_metering: FleetDataMetering;
    sms_commands_enabled: boolean;
    sms_commands_url: string;
    sms_commands_method: string;
    network_access_profile_sid: string;
    ip_commands_url: string;
    ip_commands_method: string;
}
export declare class FleetInstance {
    protected _version: V1;
    protected _solution: FleetContextSolution;
    protected _context?: FleetContext;
    constructor(_version: V1, payload: FleetResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Fleet resource.
     */
    accountSid: string;
    /**
     * The unique string that we created to identify the Fleet resource.
     */
    sid: string;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used in place of the resource\'s `sid` in the URL to address the resource.
     */
    uniqueName: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Fleet resource.
     */
    url: string;
    /**
     * Defines whether SIMs in the Fleet are capable of using 2G/3G/4G/LTE/CAT-M data connectivity. Defaults to `true`.
     */
    dataEnabled: boolean;
    /**
     * The total data usage (download and upload combined) in Megabytes that each Super SIM assigned to the Fleet can consume during a billing period (normally one month). Value must be between 1MB (1) and 2TB (2,000,000). Defaults to 250MB.
     */
    dataLimit: number;
    dataMetering: FleetDataMetering;
    /**
     * Defines whether SIMs in the Fleet are capable of sending and receiving machine-to-machine SMS via Commands. Defaults to `false`.
     */
    smsCommandsEnabled: boolean;
    /**
     * The URL that will receive a webhook when a Super SIM in the Fleet is used to send an SMS from your device to the SMS Commands number. Your server should respond with an HTTP status code in the 200 range; any response body will be ignored.
     */
    smsCommandsUrl: string;
    /**
     * A string representing the HTTP method to use when making a request to `sms_commands_url`. Can be one of `POST` or `GET`. Defaults to `POST`.
     */
    smsCommandsMethod: string;
    /**
     * The SID of the Network Access Profile that controls which cellular networks the Fleet\'s SIMs can connect to.
     */
    networkAccessProfileSid: string;
    /**
     * The URL that will receive a webhook when a Super SIM in the Fleet is used to send an IP Command from your device to a special IP address. Your server should respond with an HTTP status code in the 200 range; any response body will be ignored.
     */
    ipCommandsUrl: string;
    /**
     * A string representing the HTTP method to use when making a request to `ip_commands_url`. Can be one of `POST` or `GET`. Defaults to `POST`.
     */
    ipCommandsMethod: string;
    private get _proxy();
    /**
     * Fetch a FleetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FleetInstance
     */
    fetch(callback?: (error: Error | null, item?: FleetInstance) => any): Promise<FleetInstance>;
    /**
     * Update a FleetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FleetInstance
     */
    update(callback?: (error: Error | null, item?: FleetInstance) => any): Promise<FleetInstance>;
    /**
     * Update a FleetInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FleetInstance
     */
    update(params: FleetContextUpdateOptions, callback?: (error: Error | null, item?: FleetInstance) => any): Promise<FleetInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        sid: string;
        uniqueName: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        dataEnabled: boolean;
        dataLimit: number;
        dataMetering: "payg";
        smsCommandsEnabled: boolean;
        smsCommandsUrl: string;
        smsCommandsMethod: string;
        networkAccessProfileSid: string;
        ipCommandsUrl: string;
        ipCommandsMethod: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface FleetSolution {
}
export interface FleetListInstance {
    _version: V1;
    _solution: FleetSolution;
    _uri: string;
    (sid: string): FleetContext;
    get(sid: string): FleetContext;
    /**
     * Create a FleetInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FleetInstance
     */
    create(params: FleetListInstanceCreateOptions, callback?: (error: Error | null, item?: FleetInstance) => any): Promise<FleetInstance>;
    /**
     * Streams FleetInstance records from the API.
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
     * @param { FleetListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: FleetInstance, done: (err?: Error) => void) => void): void;
    each(params: FleetListInstanceEachOptions, callback?: (item: FleetInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of FleetInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: FleetPage) => any): Promise<FleetPage>;
    /**
     * Lists FleetInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FleetListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: FleetInstance[]) => any): Promise<FleetInstance[]>;
    list(params: FleetListInstanceOptions, callback?: (error: Error | null, items: FleetInstance[]) => any): Promise<FleetInstance[]>;
    /**
     * Retrieve a single page of FleetInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FleetListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: FleetPage) => any): Promise<FleetPage>;
    page(params: FleetListInstancePageOptions, callback?: (error: Error | null, items: FleetPage) => any): Promise<FleetPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function FleetListInstance(version: V1): FleetListInstance;
export declare class FleetPage extends Page<V1, FleetPayload, FleetResource, FleetInstance> {
    /**
     * Initialize the FleetPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: FleetSolution);
    /**
     * Build an instance of FleetInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: FleetResource): FleetInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
