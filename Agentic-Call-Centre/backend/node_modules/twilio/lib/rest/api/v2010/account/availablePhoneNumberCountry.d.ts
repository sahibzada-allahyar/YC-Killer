/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
import { LocalListInstance } from "./availablePhoneNumberCountry/local";
import { MachineToMachineListInstance } from "./availablePhoneNumberCountry/machineToMachine";
import { MobileListInstance } from "./availablePhoneNumberCountry/mobile";
import { NationalListInstance } from "./availablePhoneNumberCountry/national";
import { SharedCostListInstance } from "./availablePhoneNumberCountry/sharedCost";
import { TollFreeListInstance } from "./availablePhoneNumberCountry/tollFree";
import { VoipListInstance } from "./availablePhoneNumberCountry/voip";
/**
 * Options to pass to each
 */
export interface AvailablePhoneNumberCountryListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AvailablePhoneNumberCountryInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AvailablePhoneNumberCountryListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AvailablePhoneNumberCountryListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AvailablePhoneNumberCountryContext {
    local: LocalListInstance;
    machineToMachine: MachineToMachineListInstance;
    mobile: MobileListInstance;
    national: NationalListInstance;
    sharedCost: SharedCostListInstance;
    tollFree: TollFreeListInstance;
    voip: VoipListInstance;
    /**
     * Fetch a AvailablePhoneNumberCountryInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AvailablePhoneNumberCountryInstance
     */
    fetch(callback?: (error: Error | null, item?: AvailablePhoneNumberCountryInstance) => any): Promise<AvailablePhoneNumberCountryInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AvailablePhoneNumberCountryContextSolution {
    accountSid: string;
    countryCode: string;
}
export declare class AvailablePhoneNumberCountryContextImpl implements AvailablePhoneNumberCountryContext {
    protected _version: V2010;
    protected _solution: AvailablePhoneNumberCountryContextSolution;
    protected _uri: string;
    protected _local?: LocalListInstance;
    protected _machineToMachine?: MachineToMachineListInstance;
    protected _mobile?: MobileListInstance;
    protected _national?: NationalListInstance;
    protected _sharedCost?: SharedCostListInstance;
    protected _tollFree?: TollFreeListInstance;
    protected _voip?: VoipListInstance;
    constructor(_version: V2010, accountSid: string, countryCode: string);
    get local(): LocalListInstance;
    get machineToMachine(): MachineToMachineListInstance;
    get mobile(): MobileListInstance;
    get national(): NationalListInstance;
    get sharedCost(): SharedCostListInstance;
    get tollFree(): TollFreeListInstance;
    get voip(): VoipListInstance;
    fetch(callback?: (error: Error | null, item?: AvailablePhoneNumberCountryInstance) => any): Promise<AvailablePhoneNumberCountryInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AvailablePhoneNumberCountryContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AvailablePhoneNumberCountryPayload extends TwilioResponsePayload {
    countries: AvailablePhoneNumberCountryResource[];
}
interface AvailablePhoneNumberCountryResource {
    country_code: string;
    country: string;
    uri: string;
    beta: boolean;
    subresource_uris: Record<string, string>;
}
export declare class AvailablePhoneNumberCountryInstance {
    protected _version: V2010;
    protected _solution: AvailablePhoneNumberCountryContextSolution;
    protected _context?: AvailablePhoneNumberCountryContext;
    constructor(_version: V2010, payload: AvailablePhoneNumberCountryResource, accountSid: string, countryCode?: string);
    /**
     * The [ISO-3166-1](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) country code of the country.
     */
    countryCode: string;
    /**
     * The name of the country.
     */
    country: string;
    /**
     * The URI of the Country resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    /**
     * Whether all phone numbers available in the country are new to the Twilio platform. `true` if they are and `false` if all numbers are not in the Twilio Phone Number Beta program.
     */
    beta: boolean;
    /**
     * A list of related AvailablePhoneNumber resources identified by their URIs relative to `https://api.twilio.com`.
     */
    subresourceUris: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a AvailablePhoneNumberCountryInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AvailablePhoneNumberCountryInstance
     */
    fetch(callback?: (error: Error | null, item?: AvailablePhoneNumberCountryInstance) => any): Promise<AvailablePhoneNumberCountryInstance>;
    /**
     * Access the local.
     */
    local(): LocalListInstance;
    /**
     * Access the machineToMachine.
     */
    machineToMachine(): MachineToMachineListInstance;
    /**
     * Access the mobile.
     */
    mobile(): MobileListInstance;
    /**
     * Access the national.
     */
    national(): NationalListInstance;
    /**
     * Access the sharedCost.
     */
    sharedCost(): SharedCostListInstance;
    /**
     * Access the tollFree.
     */
    tollFree(): TollFreeListInstance;
    /**
     * Access the voip.
     */
    voip(): VoipListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        countryCode: string;
        country: string;
        uri: string;
        beta: boolean;
        subresourceUris: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AvailablePhoneNumberCountrySolution {
    accountSid: string;
}
export interface AvailablePhoneNumberCountryListInstance {
    _version: V2010;
    _solution: AvailablePhoneNumberCountrySolution;
    _uri: string;
    (countryCode: string): AvailablePhoneNumberCountryContext;
    get(countryCode: string): AvailablePhoneNumberCountryContext;
    /**
     * Streams AvailablePhoneNumberCountryInstance records from the API.
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
     * @param { AvailablePhoneNumberCountryListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AvailablePhoneNumberCountryInstance, done: (err?: Error) => void) => void): void;
    each(params: AvailablePhoneNumberCountryListInstanceEachOptions, callback?: (item: AvailablePhoneNumberCountryInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AvailablePhoneNumberCountryInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AvailablePhoneNumberCountryPage) => any): Promise<AvailablePhoneNumberCountryPage>;
    /**
     * Lists AvailablePhoneNumberCountryInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AvailablePhoneNumberCountryListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AvailablePhoneNumberCountryInstance[]) => any): Promise<AvailablePhoneNumberCountryInstance[]>;
    list(params: AvailablePhoneNumberCountryListInstanceOptions, callback?: (error: Error | null, items: AvailablePhoneNumberCountryInstance[]) => any): Promise<AvailablePhoneNumberCountryInstance[]>;
    /**
     * Retrieve a single page of AvailablePhoneNumberCountryInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AvailablePhoneNumberCountryListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AvailablePhoneNumberCountryPage) => any): Promise<AvailablePhoneNumberCountryPage>;
    page(params: AvailablePhoneNumberCountryListInstancePageOptions, callback?: (error: Error | null, items: AvailablePhoneNumberCountryPage) => any): Promise<AvailablePhoneNumberCountryPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AvailablePhoneNumberCountryListInstance(version: V2010, accountSid: string): AvailablePhoneNumberCountryListInstance;
export declare class AvailablePhoneNumberCountryPage extends Page<V2010, AvailablePhoneNumberCountryPayload, AvailablePhoneNumberCountryResource, AvailablePhoneNumberCountryInstance> {
    /**
     * Initialize the AvailablePhoneNumberCountryPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: AvailablePhoneNumberCountrySolution);
    /**
     * Build an instance of AvailablePhoneNumberCountryInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AvailablePhoneNumberCountryResource): AvailablePhoneNumberCountryInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
