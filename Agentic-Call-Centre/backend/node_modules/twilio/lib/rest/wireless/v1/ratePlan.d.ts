/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to update a RatePlanInstance
 */
export interface RatePlanContextUpdateOptions {
    /** An application-defined string that uniquely identifies the resource. It can be used in place of the resource\\\'s `sid` in the URL to address the resource. */
    uniqueName?: string;
    /** A descriptive string that you create to describe the resource. It does not have to be unique. */
    friendlyName?: string;
}
/**
 * Options to pass to create a RatePlanInstance
 */
export interface RatePlanListInstanceCreateOptions {
    /** An application-defined string that uniquely identifies the resource. It can be used in place of the resource\\\'s `sid` in the URL to address the resource. */
    uniqueName?: string;
    /** A descriptive string that you create to describe the resource. It does not have to be unique. */
    friendlyName?: string;
    /** Whether SIMs can use GPRS/3G/4G/LTE data connectivity. */
    dataEnabled?: boolean;
    /** The total data usage (download and upload combined) in Megabytes that the Network allows during one month on the home network (T-Mobile USA). The metering period begins the day of activation and ends on the same day in the following month. Can be up to 2TB and the default value is `1000`. */
    dataLimit?: number;
    /** The model used to meter data usage. Can be: `payg` and `quota-1`, `quota-10`, and `quota-50`. Learn more about the available [data metering models](https://www.twilio.com/docs/iot/wireless/api/rateplan-resource#payg-vs-quota-data-plans). */
    dataMetering?: string;
    /** Whether SIMs can make, send, and receive SMS using [Commands](https://www.twilio.com/docs/iot/wireless/api/command-resource). */
    messagingEnabled?: boolean;
    /** Deprecated. */
    voiceEnabled?: boolean;
    /** Whether SIMs can roam on networks other than the home network (T-Mobile USA) in the United States. See [national roaming](https://www.twilio.com/docs/iot/wireless/api/rateplan-resource#national-roaming). */
    nationalRoamingEnabled?: boolean;
    /** The list of services that SIMs capable of using GPRS/3G/4G/LTE data connectivity can use outside of the United States. Can contain: `data` and `messaging`. */
    internationalRoaming?: Array<string>;
    /** The total data usage (download and upload combined) in Megabytes that the Network allows during one month on non-home networks in the United States. The metering period begins the day of activation and ends on the same day in the following month. Can be up to 2TB. See [national roaming](https://www.twilio.com/docs/iot/wireless/api/rateplan-resource#national-roaming) for more info. */
    nationalRoamingDataLimit?: number;
    /** The total data usage (download and upload combined) in Megabytes that the Network allows during one month when roaming outside the United States. Can be up to 2TB. */
    internationalRoamingDataLimit?: number;
}
/**
 * Options to pass to each
 */
export interface RatePlanListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: RatePlanInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface RatePlanListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface RatePlanListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface RatePlanContext {
    /**
     * Remove a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    fetch(callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Update a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    update(callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Update a RatePlanInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    update(params: RatePlanContextUpdateOptions, callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface RatePlanContextSolution {
    sid: string;
}
export declare class RatePlanContextImpl implements RatePlanContext {
    protected _version: V1;
    protected _solution: RatePlanContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    update(params?: RatePlanContextUpdateOptions | ((error: Error | null, item?: RatePlanInstance) => any), callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): RatePlanContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface RatePlanPayload extends TwilioResponsePayload {
    rate_plans: RatePlanResource[];
}
interface RatePlanResource {
    sid: string;
    unique_name: string;
    account_sid: string;
    friendly_name: string;
    data_enabled: boolean;
    data_metering: string;
    data_limit: number;
    messaging_enabled: boolean;
    voice_enabled: boolean;
    national_roaming_enabled: boolean;
    national_roaming_data_limit: number;
    international_roaming: Array<string>;
    international_roaming_data_limit: number;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class RatePlanInstance {
    protected _version: V1;
    protected _solution: RatePlanContextSolution;
    protected _context?: RatePlanContext;
    constructor(_version: V1, payload: RatePlanResource, sid?: string);
    /**
     * The unique string that we created to identify the RatePlan resource.
     */
    sid: string;
    /**
     * An application-defined string that uniquely identifies the resource. It can be used in place of the resource\'s `sid` in the URL to address the resource.
     */
    uniqueName: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the RatePlan resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * Whether SIMs can use GPRS/3G/4G/LTE data connectivity.
     */
    dataEnabled: boolean;
    /**
     * The model used to meter data usage. Can be: `payg` and `quota-1`, `quota-10`, and `quota-50`. Learn more about the available [data metering models](https://www.twilio.com/docs/iot/wireless/api/rateplan-resource#payg-vs-quota-data-plans).
     */
    dataMetering: string;
    /**
     * The total data usage (download and upload combined) in Megabytes that the Network allows during one month on the home network (T-Mobile USA). The metering period begins the day of activation and ends on the same day in the following month. Can be up to 2TB.
     */
    dataLimit: number;
    /**
     * Whether SIMs can make, send, and receive SMS using [Commands](https://www.twilio.com/docs/iot/wireless/api/command-resource).
     */
    messagingEnabled: boolean;
    /**
     * Deprecated. Whether SIMs can make and receive voice calls.
     */
    voiceEnabled: boolean;
    /**
     * Whether SIMs can roam on networks other than the home network (T-Mobile USA) in the United States. See [national roaming](https://www.twilio.com/docs/iot/wireless/api/rateplan-resource#national-roaming).
     */
    nationalRoamingEnabled: boolean;
    /**
     * The total data usage (download and upload combined) in Megabytes that the Network allows during one month on non-home networks in the United States. The metering period begins the day of activation and ends on the same day in the following month. Can be up to 2TB.
     */
    nationalRoamingDataLimit: number;
    /**
     * The list of services that SIMs capable of using GPRS/3G/4G/LTE data connectivity can use outside of the United States. Can contain: `data` and `messaging`.
     */
    internationalRoaming: Array<string>;
    /**
     * The total data usage (download and upload combined) in Megabytes that the Network allows during one month when roaming outside the United States. Can be up to 2TB.
     */
    internationalRoamingDataLimit: number;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    fetch(callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Update a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    update(callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Update a RatePlanInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    update(params: RatePlanContextUpdateOptions, callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        uniqueName: string;
        accountSid: string;
        friendlyName: string;
        dataEnabled: boolean;
        dataMetering: string;
        dataLimit: number;
        messagingEnabled: boolean;
        voiceEnabled: boolean;
        nationalRoamingEnabled: boolean;
        nationalRoamingDataLimit: number;
        internationalRoaming: string[];
        internationalRoamingDataLimit: number;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface RatePlanSolution {
}
export interface RatePlanListInstance {
    _version: V1;
    _solution: RatePlanSolution;
    _uri: string;
    (sid: string): RatePlanContext;
    get(sid: string): RatePlanContext;
    /**
     * Create a RatePlanInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    create(callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Create a RatePlanInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RatePlanInstance
     */
    create(params: RatePlanListInstanceCreateOptions, callback?: (error: Error | null, item?: RatePlanInstance) => any): Promise<RatePlanInstance>;
    /**
     * Streams RatePlanInstance records from the API.
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
     * @param { RatePlanListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: RatePlanInstance, done: (err?: Error) => void) => void): void;
    each(params: RatePlanListInstanceEachOptions, callback?: (item: RatePlanInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of RatePlanInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: RatePlanPage) => any): Promise<RatePlanPage>;
    /**
     * Lists RatePlanInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RatePlanListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: RatePlanInstance[]) => any): Promise<RatePlanInstance[]>;
    list(params: RatePlanListInstanceOptions, callback?: (error: Error | null, items: RatePlanInstance[]) => any): Promise<RatePlanInstance[]>;
    /**
     * Retrieve a single page of RatePlanInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RatePlanListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: RatePlanPage) => any): Promise<RatePlanPage>;
    page(params: RatePlanListInstancePageOptions, callback?: (error: Error | null, items: RatePlanPage) => any): Promise<RatePlanPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RatePlanListInstance(version: V1): RatePlanListInstance;
export declare class RatePlanPage extends Page<V1, RatePlanPayload, RatePlanResource, RatePlanInstance> {
    /**
     * Initialize the RatePlanPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: RatePlanSolution);
    /**
     * Build an instance of RatePlanInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: RatePlanResource): RatePlanInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
