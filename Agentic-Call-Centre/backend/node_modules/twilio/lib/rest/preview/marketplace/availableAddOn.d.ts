/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import Marketplace from "../Marketplace";
import { AvailableAddOnExtensionListInstance } from "./availableAddOn/availableAddOnExtension";
/**
 * Options to pass to each
 */
export interface AvailableAddOnListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AvailableAddOnInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AvailableAddOnListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AvailableAddOnListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AvailableAddOnContext {
    extensions: AvailableAddOnExtensionListInstance;
    /**
     * Fetch a AvailableAddOnInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AvailableAddOnInstance
     */
    fetch(callback?: (error: Error | null, item?: AvailableAddOnInstance) => any): Promise<AvailableAddOnInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AvailableAddOnContextSolution {
    sid: string;
}
export declare class AvailableAddOnContextImpl implements AvailableAddOnContext {
    protected _version: Marketplace;
    protected _solution: AvailableAddOnContextSolution;
    protected _uri: string;
    protected _extensions?: AvailableAddOnExtensionListInstance;
    constructor(_version: Marketplace, sid: string);
    get extensions(): AvailableAddOnExtensionListInstance;
    fetch(callback?: (error: Error | null, item?: AvailableAddOnInstance) => any): Promise<AvailableAddOnInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AvailableAddOnContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AvailableAddOnPayload extends TwilioResponsePayload {
    available_add_ons: AvailableAddOnResource[];
}
interface AvailableAddOnResource {
    sid: string;
    friendly_name: string;
    description: string;
    pricing_type: string;
    configuration_schema: any;
    url: string;
    links: Record<string, string>;
}
export declare class AvailableAddOnInstance {
    protected _version: Marketplace;
    protected _solution: AvailableAddOnContextSolution;
    protected _context?: AvailableAddOnContext;
    constructor(_version: Marketplace, payload: AvailableAddOnResource, sid?: string);
    /**
     * The unique string that we created to identify the AvailableAddOn resource.
     */
    sid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * A short description of the Add-on\'s functionality.
     */
    description: string;
    /**
     * How customers are charged for using this Add-on.
     */
    pricingType: string;
    /**
     * The JSON object with the configuration that must be provided when installing a given Add-on.
     */
    configurationSchema: any;
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
     * Fetch a AvailableAddOnInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AvailableAddOnInstance
     */
    fetch(callback?: (error: Error | null, item?: AvailableAddOnInstance) => any): Promise<AvailableAddOnInstance>;
    /**
     * Access the extensions.
     */
    extensions(): AvailableAddOnExtensionListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        friendlyName: string;
        description: string;
        pricingType: string;
        configurationSchema: any;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AvailableAddOnSolution {
}
export interface AvailableAddOnListInstance {
    _version: Marketplace;
    _solution: AvailableAddOnSolution;
    _uri: string;
    (sid: string): AvailableAddOnContext;
    get(sid: string): AvailableAddOnContext;
    /**
     * Streams AvailableAddOnInstance records from the API.
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
     * @param { AvailableAddOnListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AvailableAddOnInstance, done: (err?: Error) => void) => void): void;
    each(params: AvailableAddOnListInstanceEachOptions, callback?: (item: AvailableAddOnInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AvailableAddOnInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AvailableAddOnPage) => any): Promise<AvailableAddOnPage>;
    /**
     * Lists AvailableAddOnInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AvailableAddOnListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AvailableAddOnInstance[]) => any): Promise<AvailableAddOnInstance[]>;
    list(params: AvailableAddOnListInstanceOptions, callback?: (error: Error | null, items: AvailableAddOnInstance[]) => any): Promise<AvailableAddOnInstance[]>;
    /**
     * Retrieve a single page of AvailableAddOnInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AvailableAddOnListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AvailableAddOnPage) => any): Promise<AvailableAddOnPage>;
    page(params: AvailableAddOnListInstancePageOptions, callback?: (error: Error | null, items: AvailableAddOnPage) => any): Promise<AvailableAddOnPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AvailableAddOnListInstance(version: Marketplace): AvailableAddOnListInstance;
export declare class AvailableAddOnPage extends Page<Marketplace, AvailableAddOnPayload, AvailableAddOnResource, AvailableAddOnInstance> {
    /**
     * Initialize the AvailableAddOnPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: Marketplace, response: Response<string>, solution: AvailableAddOnSolution);
    /**
     * Build an instance of AvailableAddOnInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AvailableAddOnResource): AvailableAddOnInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
