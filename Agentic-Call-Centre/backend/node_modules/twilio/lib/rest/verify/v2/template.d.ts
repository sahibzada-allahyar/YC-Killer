/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V2 from "../V2";
/**
 * Options to pass to each
 */
export interface TemplateListInstanceEachOptions {
    /** String filter used to query templates with a given friendly name. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: TemplateInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface TemplateListInstanceOptions {
    /** String filter used to query templates with a given friendly name. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface TemplateListInstancePageOptions {
    /** String filter used to query templates with a given friendly name. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface TemplateSolution {
}
export interface TemplateListInstance {
    _version: V2;
    _solution: TemplateSolution;
    _uri: string;
    /**
     * Streams TemplateInstance records from the API.
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
     * @param { TemplateListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: TemplateInstance, done: (err?: Error) => void) => void): void;
    each(params: TemplateListInstanceEachOptions, callback?: (item: TemplateInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of TemplateInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: TemplatePage) => any): Promise<TemplatePage>;
    /**
     * Lists TemplateInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TemplateListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: TemplateInstance[]) => any): Promise<TemplateInstance[]>;
    list(params: TemplateListInstanceOptions, callback?: (error: Error | null, items: TemplateInstance[]) => any): Promise<TemplateInstance[]>;
    /**
     * Retrieve a single page of TemplateInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { TemplateListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: TemplatePage) => any): Promise<TemplatePage>;
    page(params: TemplateListInstancePageOptions, callback?: (error: Error | null, items: TemplatePage) => any): Promise<TemplatePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TemplateListInstance(version: V2): TemplateListInstance;
interface TemplatePayload extends TwilioResponsePayload {
    templates: TemplateResource[];
}
interface TemplateResource {
    sid: string;
    account_sid: string;
    friendly_name: string;
    channels: Array<string>;
    translations: any;
}
export declare class TemplateInstance {
    protected _version: V2;
    constructor(_version: V2, payload: TemplateResource);
    /**
     * A 34 character string that uniquely identifies a Verification Template.
     */
    sid: string;
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * A descriptive string that you create to describe a Template. It can be up to 32 characters long.
     */
    friendlyName: string;
    /**
     * A list of channels that support the Template. Can include: sms, voice.
     */
    channels: Array<string>;
    /**
     * An object that contains the different translations of the template. Every translation is identified by the language short name and contains its respective information as the approval status, text and created/modified date.
     */
    translations: any;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        friendlyName: string;
        channels: string[];
        translations: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class TemplatePage extends Page<V2, TemplatePayload, TemplateResource, TemplateInstance> {
    /**
     * Initialize the TemplatePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: TemplateSolution);
    /**
     * Build an instance of TemplateInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: TemplateResource): TemplateInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
