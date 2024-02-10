/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import DeployedDevices from "../../DeployedDevices";
/**
 * Options to pass to update a DeviceInstance
 */
export interface DeviceContextUpdateOptions {
    /** Provides a human readable descriptive text to be assigned to this Device, up to 256 characters long. */
    friendlyName?: string;
    /** Provides an arbitrary string identifier representing a human user to be associated with this Device, up to 256 characters long. */
    identity?: string;
    /** Specifies the unique string identifier of the Deployment group that this Device is going to be associated with. */
    deploymentSid?: string;
    /**  */
    enabled?: boolean;
}
/**
 * Options to pass to create a DeviceInstance
 */
export interface DeviceListInstanceCreateOptions {
    /** Provides a unique and addressable name to be assigned to this Device, to be used in addition to SID, up to 128 characters long. */
    uniqueName?: string;
    /** Provides a human readable descriptive text to be assigned to this Device, up to 256 characters long. */
    friendlyName?: string;
    /** Provides an arbitrary string identifier representing a human user to be associated with this Device, up to 256 characters long. */
    identity?: string;
    /** Specifies the unique string identifier of the Deployment group that this Device is going to be associated with. */
    deploymentSid?: string;
    /**  */
    enabled?: boolean;
}
/**
 * Options to pass to each
 */
export interface DeviceListInstanceEachOptions {
    /** Filters the resulting list of Devices by a unique string identifier of the Deployment they are associated with. */
    deploymentSid?: string;
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
    /** Filters the resulting list of Devices by a unique string identifier of the Deployment they are associated with. */
    deploymentSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface DeviceListInstancePageOptions {
    /** Filters the resulting list of Devices by a unique string identifier of the Deployment they are associated with. */
    deploymentSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface DeviceContext {
    /**
     * Remove a DeviceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
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
    fleetSid: string;
    sid: string;
}
export declare class DeviceContextImpl implements DeviceContext {
    protected _version: DeployedDevices;
    protected _solution: DeviceContextSolution;
    protected _uri: string;
    constructor(_version: DeployedDevices, fleetSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
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
    url: string;
    unique_name: string;
    friendly_name: string;
    fleet_sid: string;
    enabled: boolean;
    account_sid: string;
    identity: string;
    deployment_sid: string;
    date_created: Date;
    date_updated: Date;
    date_authenticated: Date;
}
export declare class DeviceInstance {
    protected _version: DeployedDevices;
    protected _solution: DeviceContextSolution;
    protected _context?: DeviceContext;
    constructor(_version: DeployedDevices, payload: DeviceResource, fleetSid: string, sid?: string);
    /**
     * Contains a 34 character string that uniquely identifies this Device resource.
     */
    sid: string;
    /**
     * Contains an absolute URL for this Device resource.
     */
    url: string;
    /**
     * Contains a unique and addressable name of this Device, assigned by the developer, up to 128 characters long.
     */
    uniqueName: string;
    /**
     * Contains a human readable descriptive text for this Device, up to 256 characters long
     */
    friendlyName: string;
    /**
     * Specifies the unique string identifier of the Fleet that the given Device belongs to.
     */
    fleetSid: string;
    /**
     * Contains a boolean flag indicating whether the device is enabled or not, blocks device connectivity if set to false.
     */
    enabled: boolean;
    /**
     * Specifies the unique string identifier of the Account responsible for this Device.
     */
    accountSid: string;
    /**
     * Contains an arbitrary string identifier representing a human user associated with this Device, assigned by the developer, up to 256 characters long.
     */
    identity: string;
    /**
     * Specifies the unique string identifier of the Deployment group that this Device is associated with.
     */
    deploymentSid: string;
    /**
     * Specifies the date this Device was created, given in UTC ISO 8601 format.
     */
    dateCreated: Date;
    /**
     * Specifies the date this Device was last updated, given in UTC ISO 8601 format.
     */
    dateUpdated: Date;
    /**
     * Specifies the date this Device was last authenticated, given in UTC ISO 8601 format.
     */
    dateAuthenticated: Date;
    private get _proxy();
    /**
     * Remove a DeviceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
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
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        url: string;
        uniqueName: string;
        friendlyName: string;
        fleetSid: string;
        enabled: boolean;
        accountSid: string;
        identity: string;
        deploymentSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        dateAuthenticated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DeviceSolution {
    fleetSid: string;
}
export interface DeviceListInstance {
    _version: DeployedDevices;
    _solution: DeviceSolution;
    _uri: string;
    (sid: string): DeviceContext;
    get(sid: string): DeviceContext;
    /**
     * Create a DeviceInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceInstance
     */
    create(callback?: (error: Error | null, item?: DeviceInstance) => any): Promise<DeviceInstance>;
    /**
     * Create a DeviceInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeviceInstance
     */
    create(params: DeviceListInstanceCreateOptions, callback?: (error: Error | null, item?: DeviceInstance) => any): Promise<DeviceInstance>;
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
export declare function DeviceListInstance(version: DeployedDevices, fleetSid: string): DeviceListInstance;
export declare class DevicePage extends Page<DeployedDevices, DevicePayload, DeviceResource, DeviceInstance> {
    /**
     * Initialize the DevicePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: DeployedDevices, response: Response<string>, solution: DeviceSolution);
    /**
     * Build an instance of DeviceInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DeviceResource): DeviceInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
