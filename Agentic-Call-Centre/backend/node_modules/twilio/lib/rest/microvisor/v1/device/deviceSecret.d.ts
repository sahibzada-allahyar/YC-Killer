/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to update a DeviceSecretInstance
 */
export interface DeviceSecretContextUpdateOptions {
    /** The secret value; up to 4096 characters. */
    value: string;
}
/**
 * Options to pass to create a DeviceSecretInstance
 */
export interface DeviceSecretListInstanceCreateOptions {
    /** The secret key; up to 100 characters. */
    key: string;
    /** The secret value; up to 4096 characters. */
    value: string;
}
/**
 * Options to pass to each
 */
export interface DeviceSecretListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: DeviceSecretInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface DeviceSecretListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface DeviceSecretListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface DeviceSecretContext {
    /**
     * Remove a DeviceSecretInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a DeviceSecretInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceSecretInstance
     */
    fetch(callback?: (error: Error | null, item?: DeviceSecretInstance) => any): Promise<DeviceSecretInstance>;
    /**
     * Update a DeviceSecretInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceSecretInstance
     */
    update(params: DeviceSecretContextUpdateOptions, callback?: (error: Error | null, item?: DeviceSecretInstance) => any): Promise<DeviceSecretInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface DeviceSecretContextSolution {
    deviceSid: string;
    key: string;
}
export declare class DeviceSecretContextImpl implements DeviceSecretContext {
    protected _version: V1;
    protected _solution: DeviceSecretContextSolution;
    protected _uri: string;
    constructor(_version: V1, deviceSid: string, key: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: DeviceSecretInstance) => any): Promise<DeviceSecretInstance>;
    update(params: DeviceSecretContextUpdateOptions, callback?: (error: Error | null, item?: DeviceSecretInstance) => any): Promise<DeviceSecretInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): DeviceSecretContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface DeviceSecretPayload extends TwilioResponsePayload {
    secrets: DeviceSecretResource[];
}
interface DeviceSecretResource {
    device_sid: string;
    key: string;
    date_rotated: Date;
    url: string;
}
export declare class DeviceSecretInstance {
    protected _version: V1;
    protected _solution: DeviceSecretContextSolution;
    protected _context?: DeviceSecretContext;
    constructor(_version: V1, payload: DeviceSecretResource, deviceSid: string, key?: string);
    /**
     * A 34-character string that uniquely identifies the parent Device.
     */
    deviceSid: string;
    /**
     * The secret key; up to 100 characters.
     */
    key: string;
    dateRotated: Date;
    /**
     * The absolute URL of the Secret.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a DeviceSecretInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a DeviceSecretInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceSecretInstance
     */
    fetch(callback?: (error: Error | null, item?: DeviceSecretInstance) => any): Promise<DeviceSecretInstance>;
    /**
     * Update a DeviceSecretInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceSecretInstance
     */
    update(params: DeviceSecretContextUpdateOptions, callback?: (error: Error | null, item?: DeviceSecretInstance) => any): Promise<DeviceSecretInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        deviceSid: string;
        key: string;
        dateRotated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DeviceSecretSolution {
    deviceSid: string;
}
export interface DeviceSecretListInstance {
    _version: V1;
    _solution: DeviceSecretSolution;
    _uri: string;
    (key: string): DeviceSecretContext;
    get(key: string): DeviceSecretContext;
    /**
     * Create a DeviceSecretInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceSecretInstance
     */
    create(params: DeviceSecretListInstanceCreateOptions, callback?: (error: Error | null, item?: DeviceSecretInstance) => any): Promise<DeviceSecretInstance>;
    /**
     * Streams DeviceSecretInstance records from the API.
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
     * @param { DeviceSecretListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: DeviceSecretInstance, done: (err?: Error) => void) => void): void;
    each(params: DeviceSecretListInstanceEachOptions, callback?: (item: DeviceSecretInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of DeviceSecretInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: DeviceSecretPage) => any): Promise<DeviceSecretPage>;
    /**
     * Lists DeviceSecretInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DeviceSecretListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: DeviceSecretInstance[]) => any): Promise<DeviceSecretInstance[]>;
    list(params: DeviceSecretListInstanceOptions, callback?: (error: Error | null, items: DeviceSecretInstance[]) => any): Promise<DeviceSecretInstance[]>;
    /**
     * Retrieve a single page of DeviceSecretInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DeviceSecretListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: DeviceSecretPage) => any): Promise<DeviceSecretPage>;
    page(params: DeviceSecretListInstancePageOptions, callback?: (error: Error | null, items: DeviceSecretPage) => any): Promise<DeviceSecretPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DeviceSecretListInstance(version: V1, deviceSid: string): DeviceSecretListInstance;
export declare class DeviceSecretPage extends Page<V1, DeviceSecretPayload, DeviceSecretResource, DeviceSecretInstance> {
    /**
     * Initialize the DeviceSecretPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: DeviceSecretSolution);
    /**
     * Build an instance of DeviceSecretInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DeviceSecretResource): DeviceSecretInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
