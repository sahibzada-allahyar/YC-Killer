/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import DeployedDevices from "../../DeployedDevices";
/**
 * Options to pass to update a DeploymentInstance
 */
export interface DeploymentContextUpdateOptions {
    /** Provides a human readable descriptive text for this Deployment, up to 64 characters long */
    friendlyName?: string;
    /** Provides the unique string identifier of the Twilio Sync service instance that will be linked to and accessible by this Deployment. */
    syncServiceSid?: string;
}
/**
 * Options to pass to create a DeploymentInstance
 */
export interface DeploymentListInstanceCreateOptions {
    /** Provides a human readable descriptive text for this Deployment, up to 256 characters long. */
    friendlyName?: string;
    /** Provides the unique string identifier of the Twilio Sync service instance that will be linked to and accessible by this Deployment. */
    syncServiceSid?: string;
}
/**
 * Options to pass to each
 */
export interface DeploymentListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: DeploymentInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface DeploymentListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface DeploymentListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface DeploymentContext {
    /**
     * Remove a DeploymentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a DeploymentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeploymentInstance
     */
    fetch(callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Update a DeploymentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeploymentInstance
     */
    update(callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Update a DeploymentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeploymentInstance
     */
    update(params: DeploymentContextUpdateOptions, callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface DeploymentContextSolution {
    fleetSid: string;
    sid: string;
}
export declare class DeploymentContextImpl implements DeploymentContext {
    protected _version: DeployedDevices;
    protected _solution: DeploymentContextSolution;
    protected _uri: string;
    constructor(_version: DeployedDevices, fleetSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    update(params?: DeploymentContextUpdateOptions | ((error: Error | null, item?: DeploymentInstance) => any), callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): DeploymentContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface DeploymentPayload extends TwilioResponsePayload {
    deployments: DeploymentResource[];
}
interface DeploymentResource {
    sid: string;
    url: string;
    friendly_name: string;
    fleet_sid: string;
    account_sid: string;
    sync_service_sid: string;
    date_created: Date;
    date_updated: Date;
}
export declare class DeploymentInstance {
    protected _version: DeployedDevices;
    protected _solution: DeploymentContextSolution;
    protected _context?: DeploymentContext;
    constructor(_version: DeployedDevices, payload: DeploymentResource, fleetSid: string, sid?: string);
    /**
     * Contains a 34 character string that uniquely identifies this Deployment resource.
     */
    sid: string;
    /**
     * Contains an absolute URL for this Deployment resource.
     */
    url: string;
    /**
     * Contains a human readable descriptive text for this Deployment, up to 64 characters long
     */
    friendlyName: string;
    /**
     * Specifies the unique string identifier of the Fleet that the given Deployment belongs to.
     */
    fleetSid: string;
    /**
     * Specifies the unique string identifier of the Account responsible for this Deployment.
     */
    accountSid: string;
    /**
     * Specifies the unique string identifier of the Twilio Sync service instance linked to and accessible by this Deployment.
     */
    syncServiceSid: string;
    /**
     * Specifies the date this Deployment was created, given in UTC ISO 8601 format.
     */
    dateCreated: Date;
    /**
     * Specifies the date this Deployment was last updated, given in UTC ISO 8601 format.
     */
    dateUpdated: Date;
    private get _proxy();
    /**
     * Remove a DeploymentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a DeploymentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeploymentInstance
     */
    fetch(callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Update a DeploymentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeploymentInstance
     */
    update(callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Update a DeploymentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeploymentInstance
     */
    update(params: DeploymentContextUpdateOptions, callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        url: string;
        friendlyName: string;
        fleetSid: string;
        accountSid: string;
        syncServiceSid: string;
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DeploymentSolution {
    fleetSid: string;
}
export interface DeploymentListInstance {
    _version: DeployedDevices;
    _solution: DeploymentSolution;
    _uri: string;
    (sid: string): DeploymentContext;
    get(sid: string): DeploymentContext;
    /**
     * Create a DeploymentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeploymentInstance
     */
    create(callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Create a DeploymentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeploymentInstance
     */
    create(params: DeploymentListInstanceCreateOptions, callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Streams DeploymentInstance records from the API.
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
     * @param { DeploymentListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: DeploymentInstance, done: (err?: Error) => void) => void): void;
    each(params: DeploymentListInstanceEachOptions, callback?: (item: DeploymentInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of DeploymentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: DeploymentPage) => any): Promise<DeploymentPage>;
    /**
     * Lists DeploymentInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DeploymentListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: DeploymentInstance[]) => any): Promise<DeploymentInstance[]>;
    list(params: DeploymentListInstanceOptions, callback?: (error: Error | null, items: DeploymentInstance[]) => any): Promise<DeploymentInstance[]>;
    /**
     * Retrieve a single page of DeploymentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DeploymentListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: DeploymentPage) => any): Promise<DeploymentPage>;
    page(params: DeploymentListInstancePageOptions, callback?: (error: Error | null, items: DeploymentPage) => any): Promise<DeploymentPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DeploymentListInstance(version: DeployedDevices, fleetSid: string): DeploymentListInstance;
export declare class DeploymentPage extends Page<DeployedDevices, DeploymentPayload, DeploymentResource, DeploymentInstance> {
    /**
     * Initialize the DeploymentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: DeployedDevices, response: Response<string>, solution: DeploymentSolution);
    /**
     * Build an instance of DeploymentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DeploymentResource): DeploymentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
