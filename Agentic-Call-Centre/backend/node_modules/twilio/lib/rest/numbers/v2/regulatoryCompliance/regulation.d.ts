/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
export type RegulationEndUserType = "individual" | "business";
/**
 * Options to pass to each
 */
export interface RegulationListInstanceEachOptions {
    /** The type of End User the regulation requires - can be `individual` or `business`. */
    endUserType?: RegulationEndUserType;
    /** The ISO country code of the phone number\'s country. */
    isoCountry?: string;
    /** The type of phone number that the regulatory requiremnt is restricting. */
    numberType?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: RegulationInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface RegulationListInstanceOptions {
    /** The type of End User the regulation requires - can be `individual` or `business`. */
    endUserType?: RegulationEndUserType;
    /** The ISO country code of the phone number\'s country. */
    isoCountry?: string;
    /** The type of phone number that the regulatory requiremnt is restricting. */
    numberType?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface RegulationListInstancePageOptions {
    /** The type of End User the regulation requires - can be `individual` or `business`. */
    endUserType?: RegulationEndUserType;
    /** The ISO country code of the phone number\'s country. */
    isoCountry?: string;
    /** The type of phone number that the regulatory requiremnt is restricting. */
    numberType?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface RegulationContext {
    /**
     * Fetch a RegulationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RegulationInstance
     */
    fetch(callback?: (error: Error | null, item?: RegulationInstance) => any): Promise<RegulationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface RegulationContextSolution {
    sid: string;
}
export declare class RegulationContextImpl implements RegulationContext {
    protected _version: V2;
    protected _solution: RegulationContextSolution;
    protected _uri: string;
    constructor(_version: V2, sid: string);
    fetch(callback?: (error: Error | null, item?: RegulationInstance) => any): Promise<RegulationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): RegulationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface RegulationPayload extends TwilioResponsePayload {
    results: RegulationResource[];
}
interface RegulationResource {
    sid: string;
    friendly_name: string;
    iso_country: string;
    number_type: string;
    end_user_type: RegulationEndUserType;
    requirements: any;
    url: string;
}
export declare class RegulationInstance {
    protected _version: V2;
    protected _solution: RegulationContextSolution;
    protected _context?: RegulationContext;
    constructor(_version: V2, payload: RegulationResource, sid?: string);
    /**
     * The unique string that identifies the Regulation resource.
     */
    sid: string;
    /**
     * A human-readable description that is assigned to describe the Regulation resource. Examples can include Germany: Mobile - Business.
     */
    friendlyName: string;
    /**
     * The ISO country code of the phone number\'s country.
     */
    isoCountry: string;
    /**
     * The type of phone number restricted by the regulatory requirement. For example, Germany mobile phone numbers provisioned by businesses require a business name with commercial register proof from the Handelsregisterauszug and a proof of address from Handelsregisterauszug or a trade license by Gewerbeanmeldung.
     */
    numberType: string;
    endUserType: RegulationEndUserType;
    /**
     * The SID of an object that holds the regulatory information of the phone number country, phone number type, and end user type.
     */
    requirements: any;
    /**
     * The absolute URL of the Regulation resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a RegulationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RegulationInstance
     */
    fetch(callback?: (error: Error | null, item?: RegulationInstance) => any): Promise<RegulationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        friendlyName: string;
        isoCountry: string;
        numberType: string;
        endUserType: RegulationEndUserType;
        requirements: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface RegulationSolution {
}
export interface RegulationListInstance {
    _version: V2;
    _solution: RegulationSolution;
    _uri: string;
    (sid: string): RegulationContext;
    get(sid: string): RegulationContext;
    /**
     * Streams RegulationInstance records from the API.
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
     * @param { RegulationListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: RegulationInstance, done: (err?: Error) => void) => void): void;
    each(params: RegulationListInstanceEachOptions, callback?: (item: RegulationInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of RegulationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: RegulationPage) => any): Promise<RegulationPage>;
    /**
     * Lists RegulationInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RegulationListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: RegulationInstance[]) => any): Promise<RegulationInstance[]>;
    list(params: RegulationListInstanceOptions, callback?: (error: Error | null, items: RegulationInstance[]) => any): Promise<RegulationInstance[]>;
    /**
     * Retrieve a single page of RegulationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { RegulationListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: RegulationPage) => any): Promise<RegulationPage>;
    page(params: RegulationListInstancePageOptions, callback?: (error: Error | null, items: RegulationPage) => any): Promise<RegulationPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RegulationListInstance(version: V2): RegulationListInstance;
export declare class RegulationPage extends Page<V2, RegulationPayload, RegulationResource, RegulationInstance> {
    /**
     * Initialize the RegulationPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: RegulationSolution);
    /**
     * Build an instance of RegulationInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: RegulationResource): RegulationInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
