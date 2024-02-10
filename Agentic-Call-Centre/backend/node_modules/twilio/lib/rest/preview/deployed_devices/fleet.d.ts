/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import DeployedDevices from "../DeployedDevices";
import { CertificateListInstance } from "./fleet/certificate";
import { DeploymentListInstance } from "./fleet/deployment";
import { DeviceListInstance } from "./fleet/device";
import { KeyListInstance } from "./fleet/key";
/**
 * Options to pass to update a FleetInstance
 */
export interface FleetContextUpdateOptions {
    /** Provides a human readable descriptive text for this Fleet, up to 256 characters long. */
    friendlyName?: string;
    /** Provides a string identifier of a Deployment that is going to be used as a default one for this Fleet. */
    defaultDeploymentSid?: string;
}
/**
 * Options to pass to create a FleetInstance
 */
export interface FleetListInstanceCreateOptions {
    /** Provides a human readable descriptive text for this Fleet, up to 256 characters long. */
    friendlyName?: string;
}
/**
 * Options to pass to each
 */
export interface FleetListInstanceEachOptions {
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
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface FleetListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface FleetContext {
    certificates: CertificateListInstance;
    deployments: DeploymentListInstance;
    devices: DeviceListInstance;
    keys: KeyListInstance;
    /**
     * Remove a FleetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
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
    protected _version: DeployedDevices;
    protected _solution: FleetContextSolution;
    protected _uri: string;
    protected _certificates?: CertificateListInstance;
    protected _deployments?: DeploymentListInstance;
    protected _devices?: DeviceListInstance;
    protected _keys?: KeyListInstance;
    constructor(_version: DeployedDevices, sid: string);
    get certificates(): CertificateListInstance;
    get deployments(): DeploymentListInstance;
    get devices(): DeviceListInstance;
    get keys(): KeyListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
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
    sid: string;
    url: string;
    unique_name: string;
    friendly_name: string;
    account_sid: string;
    default_deployment_sid: string;
    date_created: Date;
    date_updated: Date;
    links: Record<string, string>;
}
export declare class FleetInstance {
    protected _version: DeployedDevices;
    protected _solution: FleetContextSolution;
    protected _context?: FleetContext;
    constructor(_version: DeployedDevices, payload: FleetResource, sid?: string);
    /**
     * Contains a 34 character string that uniquely identifies this Fleet resource.
     */
    sid: string;
    /**
     * Contains an absolute URL for this Fleet resource.
     */
    url: string;
    /**
     * Contains a unique and addressable name of this Fleet, e.g. \'default\', up to 128 characters long.
     */
    uniqueName: string;
    /**
     * Contains a human readable descriptive text for this Fleet, up to 256 characters long.
     */
    friendlyName: string;
    /**
     * Speicifies the unique string identifier of the Account responsible for this Fleet.
     */
    accountSid: string;
    /**
     * Contains the string identifier of the automatically provisioned default Deployment of this Fleet.
     */
    defaultDeploymentSid: string;
    /**
     * Specifies the date this Fleet was created, given in UTC ISO 8601 format.
     */
    dateCreated: Date;
    /**
     * Specifies the date this Fleet was last updated, given in UTC ISO 8601 format.
     */
    dateUpdated: Date;
    /**
     * Contains a dictionary of URL links to nested resources of this Fleet.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a FleetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
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
     * Access the certificates.
     */
    certificates(): CertificateListInstance;
    /**
     * Access the deployments.
     */
    deployments(): DeploymentListInstance;
    /**
     * Access the devices.
     */
    devices(): DeviceListInstance;
    /**
     * Access the keys.
     */
    keys(): KeyListInstance;
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
        accountSid: string;
        defaultDeploymentSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface FleetSolution {
}
export interface FleetListInstance {
    _version: DeployedDevices;
    _solution: FleetSolution;
    _uri: string;
    (sid: string): FleetContext;
    get(sid: string): FleetContext;
    /**
     * Create a FleetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FleetInstance
     */
    create(callback?: (error: Error | null, item?: FleetInstance) => any): Promise<FleetInstance>;
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
export declare function FleetListInstance(version: DeployedDevices): FleetListInstance;
export declare class FleetPage extends Page<DeployedDevices, FleetPayload, FleetResource, FleetInstance> {
    /**
     * Initialize the FleetPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: DeployedDevices, response: Response<string>, solution: FleetSolution);
    /**
     * Build an instance of FleetInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: FleetResource): FleetInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
