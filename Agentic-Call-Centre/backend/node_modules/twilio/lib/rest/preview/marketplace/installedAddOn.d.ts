/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import Marketplace from "../Marketplace";
import { InstalledAddOnExtensionListInstance } from "./installedAddOn/installedAddOnExtension";
/**
 * Options to pass to update a InstalledAddOnInstance
 */
export interface InstalledAddOnContextUpdateOptions {
    /** Valid JSON object that conform to the configuration schema exposed by the associated AvailableAddOn resource. This is only required by Add-ons that need to be configured */
    configuration?: any;
    /** An application-defined string that uniquely identifies the resource. This value must be unique within the Account. */
    uniqueName?: string;
}
/**
 * Options to pass to create a InstalledAddOnInstance
 */
export interface InstalledAddOnListInstanceCreateOptions {
    /** The SID of the AvaliableAddOn to install. */
    availableAddOnSid: string;
    /** Whether the Terms of Service were accepted. */
    acceptTermsOfService: boolean;
    /** The JSON object that represents the configuration of the new Add-on being installed. */
    configuration?: any;
    /** An application-defined string that uniquely identifies the resource. This value must be unique within the Account. */
    uniqueName?: string;
}
/**
 * Options to pass to each
 */
export interface InstalledAddOnListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: InstalledAddOnInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface InstalledAddOnListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface InstalledAddOnListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface InstalledAddOnContext {
    extensions: InstalledAddOnExtensionListInstance;
    /**
     * Remove a InstalledAddOnInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a InstalledAddOnInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InstalledAddOnInstance
     */
    fetch(callback?: (error: Error | null, item?: InstalledAddOnInstance) => any): Promise<InstalledAddOnInstance>;
    /**
     * Update a InstalledAddOnInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InstalledAddOnInstance
     */
    update(callback?: (error: Error | null, item?: InstalledAddOnInstance) => any): Promise<InstalledAddOnInstance>;
    /**
     * Update a InstalledAddOnInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InstalledAddOnInstance
     */
    update(params: InstalledAddOnContextUpdateOptions, callback?: (error: Error | null, item?: InstalledAddOnInstance) => any): Promise<InstalledAddOnInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface InstalledAddOnContextSolution {
    sid: string;
}
export declare class InstalledAddOnContextImpl implements InstalledAddOnContext {
    protected _version: Marketplace;
    protected _solution: InstalledAddOnContextSolution;
    protected _uri: string;
    protected _extensions?: InstalledAddOnExtensionListInstance;
    constructor(_version: Marketplace, sid: string);
    get extensions(): InstalledAddOnExtensionListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: InstalledAddOnInstance) => any): Promise<InstalledAddOnInstance>;
    update(params?: InstalledAddOnContextUpdateOptions | ((error: Error | null, item?: InstalledAddOnInstance) => any), callback?: (error: Error | null, item?: InstalledAddOnInstance) => any): Promise<InstalledAddOnInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): InstalledAddOnContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface InstalledAddOnPayload extends TwilioResponsePayload {
    installed_add_ons: InstalledAddOnResource[];
}
interface InstalledAddOnResource {
    sid: string;
    account_sid: string;
    friendly_name: string;
    description: string;
    configuration: any;
    unique_name: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class InstalledAddOnInstance {
    protected _version: Marketplace;
    protected _solution: InstalledAddOnContextSolution;
    protected _context?: InstalledAddOnContext;
    constructor(_version: Marketplace, payload: InstalledAddOnResource, sid?: string);
    /**
     * The unique string that we created to identify the InstalledAddOn resource. This Sid can also be found in the Console on that specific Add-ons page as the \'Available Add-on Sid\'.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the InstalledAddOn resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * A short description of the Add-on\'s functionality.
     */
    description: string;
    /**
     * The JSON object that represents the current configuration of installed Add-on.
     */
    configuration: any;
    /**
     * An application-defined string that uniquely identifies the resource.
     */
    uniqueName: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a InstalledAddOnInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a InstalledAddOnInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InstalledAddOnInstance
     */
    fetch(callback?: (error: Error | null, item?: InstalledAddOnInstance) => any): Promise<InstalledAddOnInstance>;
    /**
     * Update a InstalledAddOnInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InstalledAddOnInstance
     */
    update(callback?: (error: Error | null, item?: InstalledAddOnInstance) => any): Promise<InstalledAddOnInstance>;
    /**
     * Update a InstalledAddOnInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InstalledAddOnInstance
     */
    update(params: InstalledAddOnContextUpdateOptions, callback?: (error: Error | null, item?: InstalledAddOnInstance) => any): Promise<InstalledAddOnInstance>;
    /**
     * Access the extensions.
     */
    extensions(): InstalledAddOnExtensionListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        friendlyName: string;
        description: string;
        configuration: any;
        uniqueName: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface InstalledAddOnSolution {
}
export interface InstalledAddOnListInstance {
    _version: Marketplace;
    _solution: InstalledAddOnSolution;
    _uri: string;
    (sid: string): InstalledAddOnContext;
    get(sid: string): InstalledAddOnContext;
    /**
     * Create a InstalledAddOnInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InstalledAddOnInstance
     */
    create(params: InstalledAddOnListInstanceCreateOptions, callback?: (error: Error | null, item?: InstalledAddOnInstance) => any): Promise<InstalledAddOnInstance>;
    /**
     * Streams InstalledAddOnInstance records from the API.
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
     * @param { InstalledAddOnListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: InstalledAddOnInstance, done: (err?: Error) => void) => void): void;
    each(params: InstalledAddOnListInstanceEachOptions, callback?: (item: InstalledAddOnInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of InstalledAddOnInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: InstalledAddOnPage) => any): Promise<InstalledAddOnPage>;
    /**
     * Lists InstalledAddOnInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InstalledAddOnListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: InstalledAddOnInstance[]) => any): Promise<InstalledAddOnInstance[]>;
    list(params: InstalledAddOnListInstanceOptions, callback?: (error: Error | null, items: InstalledAddOnInstance[]) => any): Promise<InstalledAddOnInstance[]>;
    /**
     * Retrieve a single page of InstalledAddOnInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InstalledAddOnListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: InstalledAddOnPage) => any): Promise<InstalledAddOnPage>;
    page(params: InstalledAddOnListInstancePageOptions, callback?: (error: Error | null, items: InstalledAddOnPage) => any): Promise<InstalledAddOnPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InstalledAddOnListInstance(version: Marketplace): InstalledAddOnListInstance;
export declare class InstalledAddOnPage extends Page<Marketplace, InstalledAddOnPayload, InstalledAddOnResource, InstalledAddOnInstance> {
    /**
     * Initialize the InstalledAddOnPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: Marketplace, response: Response<string>, solution: InstalledAddOnSolution);
    /**
     * Build an instance of InstalledAddOnInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: InstalledAddOnResource): InstalledAddOnInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
