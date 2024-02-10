/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to each
 */
export interface DataSessionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: DataSessionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface DataSessionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface DataSessionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface DataSessionSolution {
    simSid: string;
}
export interface DataSessionListInstance {
    _version: V1;
    _solution: DataSessionSolution;
    _uri: string;
    /**
     * Streams DataSessionInstance records from the API.
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
     * @param { DataSessionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: DataSessionInstance, done: (err?: Error) => void) => void): void;
    each(params: DataSessionListInstanceEachOptions, callback?: (item: DataSessionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of DataSessionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: DataSessionPage) => any): Promise<DataSessionPage>;
    /**
     * Lists DataSessionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DataSessionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: DataSessionInstance[]) => any): Promise<DataSessionInstance[]>;
    list(params: DataSessionListInstanceOptions, callback?: (error: Error | null, items: DataSessionInstance[]) => any): Promise<DataSessionInstance[]>;
    /**
     * Retrieve a single page of DataSessionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DataSessionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: DataSessionPage) => any): Promise<DataSessionPage>;
    page(params: DataSessionListInstancePageOptions, callback?: (error: Error | null, items: DataSessionPage) => any): Promise<DataSessionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DataSessionListInstance(version: V1, simSid: string): DataSessionListInstance;
interface DataSessionPayload extends TwilioResponsePayload {
    data_sessions: DataSessionResource[];
}
interface DataSessionResource {
    sid: string;
    sim_sid: string;
    account_sid: string;
    radio_link: string;
    operator_mcc: string;
    operator_mnc: string;
    operator_country: string;
    operator_name: string;
    cell_id: string;
    cell_location_estimate: any;
    packets_uploaded: number;
    packets_downloaded: number;
    last_updated: Date;
    start: Date;
    end: Date;
    imei: string;
}
export declare class DataSessionInstance {
    protected _version: V1;
    constructor(_version: V1, payload: DataSessionResource, simSid: string);
    /**
     * The unique string that we created to identify the DataSession resource.
     */
    sid: string;
    /**
     * The SID of the [Sim resource](https://www.twilio.com/docs/iot/wireless/api/sim-resource) that the Data Session is for.
     */
    simSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the DataSession resource.
     */
    accountSid: string;
    /**
     * The generation of wireless technology that the device was using.
     */
    radioLink: string;
    /**
     * The \'mobile country code\' is the unique ID of the home country where the Data Session took place. See: [MCC/MNC lookup](http://mcc-mnc.com/).
     */
    operatorMcc: string;
    /**
     * The \'mobile network code\' is the unique ID specific to the mobile operator network where the Data Session took place.
     */
    operatorMnc: string;
    /**
     * The three letter country code representing where the device\'s Data Session took place. This is determined by looking up the `operator_mcc`.
     */
    operatorCountry: string;
    /**
     * The friendly name of the mobile operator network that the [SIM](https://www.twilio.com/docs/iot/wireless/api/sim-resource)-connected device is attached to. This is determined by looking up the `operator_mnc`.
     */
    operatorName: string;
    /**
     * The unique ID of the cellular tower that the device was attached to at the moment when the Data Session was last updated.
     */
    cellId: string;
    /**
     * An object that describes the estimated location in latitude and longitude where the device\'s Data Session took place. The location is derived from the `cell_id` when the Data Session was last updated. See [Cell Location Estimate Object](https://www.twilio.com/docs/iot/wireless/api/datasession-resource#cell-location-estimate-object).
     */
    cellLocationEstimate: any;
    /**
     * The number of packets uploaded by the device between the `start` time and when the Data Session was last updated.
     */
    packetsUploaded: number;
    /**
     * The number of packets downloaded by the device between the `start` time and when the Data Session was last updated.
     */
    packetsDownloaded: number;
    /**
     * The date that the resource was last updated, given as GMT in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format.
     */
    lastUpdated: Date;
    /**
     * The date that the Data Session started, given as GMT in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format.
     */
    start: Date;
    /**
     * The date that the record ended, given as GMT in [ISO 8601](https://www.iso.org/iso-8601-date-and-time-format.html) format.
     */
    end: Date;
    /**
     * The \'international mobile equipment identity\' is the unique ID of the device using the SIM to connect. An IMEI is a 15-digit string: 14 digits for the device identifier plus a check digit calculated using the Luhn formula.
     */
    imei: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        simSid: string;
        accountSid: string;
        radioLink: string;
        operatorMcc: string;
        operatorMnc: string;
        operatorCountry: string;
        operatorName: string;
        cellId: string;
        cellLocationEstimate: any;
        packetsUploaded: number;
        packetsDownloaded: number;
        lastUpdated: Date;
        start: Date;
        end: Date;
        imei: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class DataSessionPage extends Page<V1, DataSessionPayload, DataSessionResource, DataSessionInstance> {
    /**
     * Initialize the DataSessionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: DataSessionSolution);
    /**
     * Build an instance of DataSessionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DataSessionResource): DataSessionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
