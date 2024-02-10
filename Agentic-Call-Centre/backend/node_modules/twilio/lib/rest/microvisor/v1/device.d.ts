/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { DeviceConfigListInstance } from "./device/deviceConfig";
import { DeviceSecretListInstance } from "./device/deviceSecret";
/**
 * Options to pass to update a DeviceInstance
 */
export interface DeviceContextUpdateOptions {
    /** A unique and addressable name to be assigned to this Device by the developer. It may be used in place of the Device SID. */
    uniqueName?: string;
    /** The SID or unique name of the App to be targeted to the Device. */
    targetApp?: string;
    /** A Boolean flag specifying whether to enable application logging. Logs will be enabled or extended for 24 hours. */
    loggingEnabled?: boolean;
    /** Set to true to restart the App running on the Device. */
    restartApp?: boolean;
}
/**
 * Options to pass to each
 */
export interface DeviceListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: DeviceInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface DeviceListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface DeviceListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface DeviceContext {
    deviceConfigs: DeviceConfigListInstance;
    deviceSecrets: DeviceSecretListInstance;
    /**
     * Fetch a DeviceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceInstance
     */
    fetch(callback?: (error: Error | null, item?: DeviceInstance) => any): Promise<DeviceInstance>;
    /**
     * Update a DeviceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceInstance
     */
    update(callback?: (error: Error | null, item?: DeviceInstance) => any): Promise<DeviceInstance>;
    /**
     * Update a DeviceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceInstance
     */
    update(params: DeviceContextUpdateOptions, callback?: (error: Error | null, item?: DeviceInstance) => any): Promise<DeviceInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface DeviceContextSolution {
    sid: string;
}
export declare class DeviceContextImpl implements DeviceContext {
    protected _version: V1;
    protected _solution: DeviceContextSolution;
    protected _uri: string;
    protected _deviceConfigs?: DeviceConfigListInstance;
    protected _deviceSecrets?: DeviceSecretListInstance;
    constructor(_version: V1, sid: string);
    get deviceConfigs(): DeviceConfigListInstance;
    get deviceSecrets(): DeviceSecretListInstance;
    fetch(callback?: (error: Error | null, item?: DeviceInstance) => any): Promise<DeviceInstance>;
    update(params?: DeviceContextUpdateOptions | ((error: Error | null, item?: DeviceInstance) => any), callback?: (error: Error | null, item?: DeviceInstance) => any): Promise<DeviceInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): DeviceContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface DevicePayload extends TwilioResponsePayload {
    devices: DeviceResource[];
}
interface DeviceResource {
    sid: string;
    unique_name: string;
    account_sid: string;
    app: any;
    logging: any;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class DeviceInstance {
    protected _version: V1;
    protected _solution: DeviceContextSolution;
    protected _context?: DeviceContext;
    constructor(_version: V1, payload: DeviceResource, sid?: string);
    /**
     * A 34-character string that uniquely identifies this Device.
     */
    sid: string;
    /**
     * A developer-defined string that uniquely identifies the Device. This value must be unique for all Devices on this Account. The `unique_name` value may be used as an alternative to the `sid` in the URL path to address the resource.
     */
    uniqueName: string;
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * Information about the target App and the App reported by this Device. Contains the properties `target_sid`, `date_targeted`, `update_status` (one of `up-to-date`, `pending` and `error`), `update_error_code`, `reported_sid` and `date_reported`.
     */
    app: any;
    /**
     * Object specifying whether application logging is enabled for this Device. Contains the properties `enabled` and `date_expires`.
     */
    logging: any;
    /**
     * The date that this Device was created, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date that this Device was last updated, given in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The URL of this resource.
     */
    url: string;
    /**
     * The absolute URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a DeviceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceInstance
     */
    fetch(callback?: (error: Error | null, item?: DeviceInstance) => any): Promise<DeviceInstance>;
    /**
     * Update a DeviceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceInstance
     */
    update(callback?: (error: Error | null, item?: DeviceInstance) => any): Promise<DeviceInstance>;
    /**
     * Update a DeviceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceInstance
     */
    update(params: DeviceContextUpdateOptions, callback?: (error: Error | null, item?: DeviceInstance) => any): Promise<DeviceInstance>;
    /**
     * Access the deviceConfigs.
     */
    deviceConfigs(): DeviceConfigListInstance;
    /**
     * Access the deviceSecrets.
     */
    deviceSecrets(): DeviceSecretListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        uniqueName: string;
        accountSid: string;
        app: any;
        logging: any;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DeviceSolution {
}
export interface DeviceListInstance {
    _version: V1;
    _solution: DeviceSolution;
    _uri: string;
    (sid: string): DeviceContext;
    get(sid: string): DeviceContext;
    /**
     * Streams DeviceInstance records from the API.
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
     * @param { DeviceListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: DeviceInstance, done: (err?: Error) => void) => void): void;
    each(params: DeviceListInstanceEachOptions, callback?: (item: DeviceInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of DeviceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: DevicePage) => any): Promise<DevicePage>;
    /**
     * Lists DeviceInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DeviceListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: DeviceInstance[]) => any): Promise<DeviceInstance[]>;
    list(params: DeviceListInstanceOptions, callback?: (error: Error | null, items: DeviceInstance[]) => any): Promise<DeviceInstance[]>;
    /**
     * Retrieve a single page of DeviceInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DeviceListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: DevicePage) => any): Promise<DevicePage>;
    page(params: DeviceListInstancePageOptions, callback?: (error: Error | null, items: DevicePage) => any): Promise<DevicePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DeviceListInstance(version: V1): DeviceListInstance;
export declare class DevicePage extends Page<V1, DevicePayload, DeviceResource, DeviceInstance> {
    /**
     * Initialize the DevicePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: DeviceSolution);
    /**
     * Build an instance of DeviceInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DeviceResource): DeviceInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
