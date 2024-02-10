/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import DeployedDevices from "../../DeployedDevices";
/**
 * Options to pass to update a KeyInstance
 */
export interface KeyContextUpdateOptions {
    /** Provides a human readable descriptive text for this Key credential, up to 256 characters long. */
    friendlyName?: string;
    /** Provides the unique string identifier of an existing Device to become authenticated with this Key credential. */
    deviceSid?: string;
}
/**
 * Options to pass to create a KeyInstance
 */
export interface KeyListInstanceCreateOptions {
    /** Provides a human readable descriptive text for this Key credential, up to 256 characters long. */
    friendlyName?: string;
    /** Provides the unique string identifier of an existing Device to become authenticated with this Key credential. */
    deviceSid?: string;
}
/**
 * Options to pass to each
 */
export interface KeyListInstanceEachOptions {
    /** Filters the resulting list of Keys by a unique string identifier of an authenticated Device. */
    deviceSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: KeyInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface KeyListInstanceOptions {
    /** Filters the resulting list of Keys by a unique string identifier of an authenticated Device. */
    deviceSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface KeyListInstancePageOptions {
    /** Filters the resulting list of Keys by a unique string identifier of an authenticated Device. */
    deviceSid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface KeyContext {
    /**
     * Remove a KeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a KeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed KeyInstance
     */
    fetch(callback?: (error: Error | null, item?: KeyInstance) => any): Promise<KeyInstance>;
    /**
     * Update a KeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed KeyInstance
     */
    update(callback?: (error: Error | null, item?: KeyInstance) => any): Promise<KeyInstance>;
    /**
     * Update a KeyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed KeyInstance
     */
    update(params: KeyContextUpdateOptions, callback?: (error: Error | null, item?: KeyInstance) => any): Promise<KeyInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface KeyContextSolution {
    fleetSid: string;
    sid: string;
}
export declare class KeyContextImpl implements KeyContext {
    protected _version: DeployedDevices;
    protected _solution: KeyContextSolution;
    protected _uri: string;
    constructor(_version: DeployedDevices, fleetSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: KeyInstance) => any): Promise<KeyInstance>;
    update(params?: KeyContextUpdateOptions | ((error: Error | null, item?: KeyInstance) => any), callback?: (error: Error | null, item?: KeyInstance) => any): Promise<KeyInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): KeyContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface KeyPayload extends TwilioResponsePayload {
    keys: KeyResource[];
}
interface KeyResource {
    sid: string;
    url: string;
    friendly_name: string;
    fleet_sid: string;
    account_sid: string;
    device_sid: string;
    secret: string;
    date_created: Date;
    date_updated: Date;
}
export declare class KeyInstance {
    protected _version: DeployedDevices;
    protected _solution: KeyContextSolution;
    protected _context?: KeyContext;
    constructor(_version: DeployedDevices, payload: KeyResource, fleetSid: string, sid?: string);
    /**
     * Contains a 34 character string that uniquely identifies this Key credential resource.
     */
    sid: string;
    /**
     * Contains an absolute URL for this Key credential resource.
     */
    url: string;
    /**
     * Contains a human readable descriptive text for this Key credential, up to 256 characters long.
     */
    friendlyName: string;
    /**
     * Specifies the unique string identifier of the Fleet that the given Key credential belongs to.
     */
    fleetSid: string;
    /**
     * Specifies the unique string identifier of the Account responsible for this Key credential.
     */
    accountSid: string;
    /**
     * Specifies the unique string identifier of a Device authenticated with this Key credential.
     */
    deviceSid: string;
    /**
     * Contains the automatically generated secret belonging to this Key credential, used to authenticate the Device.
     */
    secret: string;
    /**
     * Specifies the date this Key credential was created, given in UTC ISO 8601 format.
     */
    dateCreated: Date;
    /**
     * Specifies the date this Key credential was last updated, given in UTC ISO 8601 format.
     */
    dateUpdated: Date;
    private get _proxy();
    /**
     * Remove a KeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a KeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed KeyInstance
     */
    fetch(callback?: (error: Error | null, item?: KeyInstance) => any): Promise<KeyInstance>;
    /**
     * Update a KeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed KeyInstance
     */
    update(callback?: (error: Error | null, item?: KeyInstance) => any): Promise<KeyInstance>;
    /**
     * Update a KeyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed KeyInstance
     */
    update(params: KeyContextUpdateOptions, callback?: (error: Error | null, item?: KeyInstance) => any): Promise<KeyInstance>;
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
        deviceSid: string;
        secret: string;
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface KeySolution {
    fleetSid: string;
}
export interface KeyListInstance {
    _version: DeployedDevices;
    _solution: KeySolution;
    _uri: string;
    (sid: string): KeyContext;
    get(sid: string): KeyContext;
    /**
     * Create a KeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed KeyInstance
     */
    create(callback?: (error: Error | null, item?: KeyInstance) => any): Promise<KeyInstance>;
    /**
     * Create a KeyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed KeyInstance
     */
    create(params: KeyListInstanceCreateOptions, callback?: (error: Error | null, item?: KeyInstance) => any): Promise<KeyInstance>;
    /**
     * Streams KeyInstance records from the API.
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
     * @param { KeyListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: KeyInstance, done: (err?: Error) => void) => void): void;
    each(params: KeyListInstanceEachOptions, callback?: (item: KeyInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of KeyInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: KeyPage) => any): Promise<KeyPage>;
    /**
     * Lists KeyInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { KeyListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: KeyInstance[]) => any): Promise<KeyInstance[]>;
    list(params: KeyListInstanceOptions, callback?: (error: Error | null, items: KeyInstance[]) => any): Promise<KeyInstance[]>;
    /**
     * Retrieve a single page of KeyInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { KeyListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: KeyPage) => any): Promise<KeyPage>;
    page(params: KeyListInstancePageOptions, callback?: (error: Error | null, items: KeyPage) => any): Promise<KeyPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function KeyListInstance(version: DeployedDevices, fleetSid: string): KeyListInstance;
export declare class KeyPage extends Page<DeployedDevices, KeyPayload, KeyResource, KeyInstance> {
    /**
     * Initialize the KeyPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: DeployedDevices, response: Response<string>, solution: KeySolution);
    /**
     * Build an instance of KeyInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: KeyResource): KeyInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
