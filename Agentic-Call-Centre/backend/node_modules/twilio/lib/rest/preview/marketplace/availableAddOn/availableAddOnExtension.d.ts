/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import Marketplace from "../../Marketplace";
/**
 * Options to pass to each
 */
export interface AvailableAddOnExtensionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AvailableAddOnExtensionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AvailableAddOnExtensionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AvailableAddOnExtensionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AvailableAddOnExtensionContext {
    /**
     * Fetch a AvailableAddOnExtensionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AvailableAddOnExtensionInstance
     */
    fetch(callback?: (error: Error | null, item?: AvailableAddOnExtensionInstance) => any): Promise<AvailableAddOnExtensionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AvailableAddOnExtensionContextSolution {
    availableAddOnSid: string;
    sid: string;
}
export declare class AvailableAddOnExtensionContextImpl implements AvailableAddOnExtensionContext {
    protected _version: Marketplace;
    protected _solution: AvailableAddOnExtensionContextSolution;
    protected _uri: string;
    constructor(_version: Marketplace, availableAddOnSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: AvailableAddOnExtensionInstance) => any): Promise<AvailableAddOnExtensionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AvailableAddOnExtensionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AvailableAddOnExtensionPayload extends TwilioResponsePayload {
    extensions: AvailableAddOnExtensionResource[];
}
interface AvailableAddOnExtensionResource {
    sid: string;
    available_add_on_sid: string;
    friendly_name: string;
    product_name: string;
    unique_name: string;
    url: string;
}
export declare class AvailableAddOnExtensionInstance {
    protected _version: Marketplace;
    protected _solution: AvailableAddOnExtensionContextSolution;
    protected _context?: AvailableAddOnExtensionContext;
    constructor(_version: Marketplace, payload: AvailableAddOnExtensionResource, availableAddOnSid: string, sid?: string);
    /**
     * The unique string that we created to identify the AvailableAddOnExtension resource.
     */
    sid: string;
    /**
     * The SID of the AvailableAddOn resource to which this extension applies.
     */
    availableAddOnSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The name of the Product this Extension is used within.
     */
    productName: string;
    /**
     * An application-defined string that uniquely identifies the resource.
     */
    uniqueName: string;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a AvailableAddOnExtensionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AvailableAddOnExtensionInstance
     */
    fetch(callback?: (error: Error | null, item?: AvailableAddOnExtensionInstance) => any): Promise<AvailableAddOnExtensionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        availableAddOnSid: string;
        friendlyName: string;
        productName: string;
        uniqueName: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AvailableAddOnExtensionSolution {
    availableAddOnSid: string;
}
export interface AvailableAddOnExtensionListInstance {
    _version: Marketplace;
    _solution: AvailableAddOnExtensionSolution;
    _uri: string;
    (sid: string): AvailableAddOnExtensionContext;
    get(sid: string): AvailableAddOnExtensionContext;
    /**
     * Streams AvailableAddOnExtensionInstance records from the API.
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
     * @param { AvailableAddOnExtensionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AvailableAddOnExtensionInstance, done: (err?: Error) => void) => void): void;
    each(params: AvailableAddOnExtensionListInstanceEachOptions, callback?: (item: AvailableAddOnExtensionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AvailableAddOnExtensionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AvailableAddOnExtensionPage) => any): Promise<AvailableAddOnExtensionPage>;
    /**
     * Lists AvailableAddOnExtensionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AvailableAddOnExtensionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AvailableAddOnExtensionInstance[]) => any): Promise<AvailableAddOnExtensionInstance[]>;
    list(params: AvailableAddOnExtensionListInstanceOptions, callback?: (error: Error | null, items: AvailableAddOnExtensionInstance[]) => any): Promise<AvailableAddOnExtensionInstance[]>;
    /**
     * Retrieve a single page of AvailableAddOnExtensionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AvailableAddOnExtensionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AvailableAddOnExtensionPage) => any): Promise<AvailableAddOnExtensionPage>;
    page(params: AvailableAddOnExtensionListInstancePageOptions, callback?: (error: Error | null, items: AvailableAddOnExtensionPage) => any): Promise<AvailableAddOnExtensionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AvailableAddOnExtensionListInstance(version: Marketplace, availableAddOnSid: string): AvailableAddOnExtensionListInstance;
export declare class AvailableAddOnExtensionPage extends Page<Marketplace, AvailableAddOnExtensionPayload, AvailableAddOnExtensionResource, AvailableAddOnExtensionInstance> {
    /**
     * Initialize the AvailableAddOnExtensionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: Marketplace, response: Response<string>, solution: AvailableAddOnExtensionSolution);
    /**
     * Build an instance of AvailableAddOnExtensionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AvailableAddOnExtensionResource): AvailableAddOnExtensionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
