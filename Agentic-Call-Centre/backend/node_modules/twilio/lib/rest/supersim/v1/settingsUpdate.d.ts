/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type SettingsUpdateStatus = "scheduled" | "in-progress" | "successful" | "failed";
/**
 * Options to pass to each
 */
export interface SettingsUpdateListInstanceEachOptions {
    /** Filter the Settings Updates by a Super SIM\'s SID or UniqueName. */
    sim?: string;
    /** Filter the Settings Updates by status. Can be `scheduled`, `in-progress`, `successful`, or `failed`. */
    status?: SettingsUpdateStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SettingsUpdateInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SettingsUpdateListInstanceOptions {
    /** Filter the Settings Updates by a Super SIM\'s SID or UniqueName. */
    sim?: string;
    /** Filter the Settings Updates by status. Can be `scheduled`, `in-progress`, `successful`, or `failed`. */
    status?: SettingsUpdateStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SettingsUpdateListInstancePageOptions {
    /** Filter the Settings Updates by a Super SIM\'s SID or UniqueName. */
    sim?: string;
    /** Filter the Settings Updates by status. Can be `scheduled`, `in-progress`, `successful`, or `failed`. */
    status?: SettingsUpdateStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SettingsUpdateSolution {
}
export interface SettingsUpdateListInstance {
    _version: V1;
    _solution: SettingsUpdateSolution;
    _uri: string;
    /**
     * Streams SettingsUpdateInstance records from the API.
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
     * @param { SettingsUpdateListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SettingsUpdateInstance, done: (err?: Error) => void) => void): void;
    each(params: SettingsUpdateListInstanceEachOptions, callback?: (item: SettingsUpdateInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SettingsUpdateInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SettingsUpdatePage) => any): Promise<SettingsUpdatePage>;
    /**
     * Lists SettingsUpdateInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SettingsUpdateListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SettingsUpdateInstance[]) => any): Promise<SettingsUpdateInstance[]>;
    list(params: SettingsUpdateListInstanceOptions, callback?: (error: Error | null, items: SettingsUpdateInstance[]) => any): Promise<SettingsUpdateInstance[]>;
    /**
     * Retrieve a single page of SettingsUpdateInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SettingsUpdateListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SettingsUpdatePage) => any): Promise<SettingsUpdatePage>;
    page(params: SettingsUpdateListInstancePageOptions, callback?: (error: Error | null, items: SettingsUpdatePage) => any): Promise<SettingsUpdatePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SettingsUpdateListInstance(version: V1): SettingsUpdateListInstance;
interface SettingsUpdatePayload extends TwilioResponsePayload {
    settings_updates: SettingsUpdateResource[];
}
interface SettingsUpdateResource {
    sid: string;
    iccid: string;
    sim_sid: string;
    status: SettingsUpdateStatus;
    packages: Array<any>;
    date_completed: Date;
    date_created: Date;
    date_updated: Date;
}
export declare class SettingsUpdateInstance {
    protected _version: V1;
    constructor(_version: V1, payload: SettingsUpdateResource);
    /**
     * The unique identifier of this Settings Update.
     */
    sid: string;
    /**
     * The [ICCID](https://en.wikipedia.org/wiki/SIM_card#ICCID) associated with the SIM.
     */
    iccid: string;
    /**
     * The SID of the Super SIM to which this Settings Update was applied.
     */
    simSid: string;
    status: SettingsUpdateStatus;
    /**
     * Array containing the different Settings Packages that will be applied to the SIM after the update completes. Each object within the array indicates the name and the version of the Settings Package that will be on the SIM if the update is successful.
     */
    packages: Array<any>;
    /**
     * The time, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format, when the update successfully completed and the new settings were applied to the SIM.
     */
    dateCompleted: Date;
    /**
     * The date that this Settings Update was created, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date that this Settings Update was updated, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        iccid: string;
        simSid: string;
        status: SettingsUpdateStatus;
        packages: any[];
        dateCompleted: Date;
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class SettingsUpdatePage extends Page<V1, SettingsUpdatePayload, SettingsUpdateResource, SettingsUpdateInstance> {
    /**
     * Initialize the SettingsUpdatePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SettingsUpdateSolution);
    /**
     * Build an instance of SettingsUpdateInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SettingsUpdateResource): SettingsUpdateInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
