/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to update a DeviceConfigInstance
 */
export interface DeviceConfigContextUpdateOptions {
    /** The config value; up to 4096 characters. */
    value: string;
}
/**
 * Options to pass to create a DeviceConfigInstance
 */
export interface DeviceConfigListInstanceCreateOptions {
    /** The config key; up to 100 characters. */
    key: string;
    /** The config value; up to 4096 characters. */
    value: string;
}
/**
 * Options to pass to each
 */
export interface DeviceConfigListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: DeviceConfigInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface DeviceConfigListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface DeviceConfigListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface DeviceConfigContext {
    /**
     * Remove a DeviceConfigInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a DeviceConfigInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceConfigInstance
     */
    fetch(callback?: (error: Error | null, item?: DeviceConfigInstance) => any): Promise<DeviceConfigInstance>;
    /**
     * Update a DeviceConfigInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceConfigInstance
     */
    update(params: DeviceConfigContextUpdateOptions, callback?: (error: Error | null, item?: DeviceConfigInstance) => any): Promise<DeviceConfigInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface DeviceConfigContextSolution {
    deviceSid: string;
    key: string;
}
export declare class DeviceConfigContextImpl implements DeviceConfigContext {
    protected _version: V1;
    protected _solution: DeviceConfigContextSolution;
    protected _uri: string;
    constructor(_version: V1, deviceSid: string, key: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: DeviceConfigInstance) => any): Promise<DeviceConfigInstance>;
    update(params: DeviceConfigContextUpdateOptions, callback?: (error: Error | null, item?: DeviceConfigInstance) => any): Promise<DeviceConfigInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): DeviceConfigContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface DeviceConfigPayload extends TwilioResponsePayload {
    configs: DeviceConfigResource[];
}
interface DeviceConfigResource {
    device_sid: string;
    key: string;
    value: string;
    date_updated: Date;
    url: string;
}
export declare class DeviceConfigInstance {
    protected _version: V1;
    protected _solution: DeviceConfigContextSolution;
    protected _context?: DeviceConfigContext;
    constructor(_version: V1, payload: DeviceConfigResource, deviceSid: string, key?: string);
    /**
     * A 34-character string that uniquely identifies the parent Device.
     */
    deviceSid: string;
    /**
     * The config key; up to 100 characters.
     */
    key: string;
    /**
     * The config value; up to 4096 characters.
     */
    value: string;
    dateUpdated: Date;
    /**
     * The absolute URL of the Config.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a DeviceConfigInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a DeviceConfigInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceConfigInstance
     */
    fetch(callback?: (error: Error | null, item?: DeviceConfigInstance) => any): Promise<DeviceConfigInstance>;
    /**
     * Update a DeviceConfigInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceConfigInstance
     */
    update(params: DeviceConfigContextUpdateOptions, callback?: (error: Error | null, item?: DeviceConfigInstance) => any): Promise<DeviceConfigInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        deviceSid: string;
        key: string;
        value: string;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DeviceConfigSolution {
    deviceSid: string;
}
export interface DeviceConfigListInstance {
    _version: V1;
    _solution: DeviceConfigSolution;
    _uri: string;
    (key: string): DeviceConfigContext;
    get(key: string): DeviceConfigContext;
    /**
     * Create a DeviceConfigInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceConfigInstance
     */
    create(params: DeviceConfigListInstanceCreateOptions, callback?: (error: Error | null, item?: DeviceConfigInstance) => any): Promise<DeviceConfigInstance>;
    /**
     * Streams DeviceConfigInstance records from the API.
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
     * @param { DeviceConfigListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: DeviceConfigInstance, done: (err?: Error) => void) => void): void;
    each(params: DeviceConfigListInstanceEachOptions, callback?: (item: DeviceConfigInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of DeviceConfigInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: DeviceConfigPage) => any): Promise<DeviceConfigPage>;
    /**
     * Lists DeviceConfigInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DeviceConfigListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: DeviceConfigInstance[]) => any): Promise<DeviceConfigInstance[]>;
    list(params: DeviceConfigListInstanceOptions, callback?: (error: Error | null, items: DeviceConfigInstance[]) => any): Promise<DeviceConfigInstance[]>;
    /**
     * Retrieve a single page of DeviceConfigInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DeviceConfigListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: DeviceConfigPage) => any): Promise<DeviceConfigPage>;
    page(params: DeviceConfigListInstancePageOptions, callback?: (error: Error | null, items: DeviceConfigPage) => any): Promise<DeviceConfigPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DeviceConfigListInstance(version: V1, deviceSid: string): DeviceConfigListInstance;
export declare class DeviceConfigPage extends Page<V1, DeviceConfigPayload, DeviceConfigResource, DeviceConfigInstance> {
    /**
     * Initialize the DeviceConfigPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: DeviceConfigSolution);
    /**
     * Build an instance of DeviceConfigInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DeviceConfigResource): DeviceConfigInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
