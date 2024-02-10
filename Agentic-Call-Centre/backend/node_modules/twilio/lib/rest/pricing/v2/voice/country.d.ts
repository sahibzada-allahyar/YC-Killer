/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
export declare class PricingV2TrunkingCountryInstanceOriginatingCallPrices {
    "basePrice"?: number;
    "currentPrice"?: number;
    "numberType"?: string;
}
export declare class PricingV2TrunkingCountryInstanceTerminatingPrefixPrices {
    "originationPrefixes"?: Array<string>;
    "destinationPrefixes"?: Array<string>;
    "basePrice"?: number;
    "currentPrice"?: number;
    "friendlyName"?: string;
}
/**
 * Options to pass to each
 */
export interface CountryListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: CountryInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface CountryListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CountryListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CountryContext {
    /**
     * Fetch a CountryInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CountryInstance
     */
    fetch(callback?: (error: Error | null, item?: CountryInstance) => any): Promise<CountryInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface CountryContextSolution {
    isoCountry: string;
}
export declare class CountryContextImpl implements CountryContext {
    protected _version: V2;
    protected _solution: CountryContextSolution;
    protected _uri: string;
    constructor(_version: V2, isoCountry: string);
    fetch(callback?: (error: Error | null, item?: CountryInstance) => any): Promise<CountryInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): CountryContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface CountryPayload extends TwilioResponsePayload {
    countries: CountryResource[];
}
interface CountryResource {
    country: string;
    iso_country: string;
    outbound_prefix_prices: Array<PricingV2TrunkingCountryInstanceTerminatingPrefixPrices>;
    inbound_call_prices: Array<PricingV2TrunkingCountryInstanceOriginatingCallPrices>;
    price_unit: string;
    url: string;
}
export declare class CountryInstance {
    protected _version: V2;
    protected _solution: CountryContextSolution;
    protected _context?: CountryContext;
    constructor(_version: V2, payload: CountryResource, isoCountry?: string);
    /**
     * The name of the country.
     */
    country: string;
    /**
     * The [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
     */
    isoCountry: string;
    /**
     * The list of [OutboundPrefixPriceWithOrigin](https://www.twilio.com/docs/voice/pricing#outbound-prefix-price-with-origin) records.
     */
    outboundPrefixPrices: Array<PricingV2TrunkingCountryInstanceTerminatingPrefixPrices>;
    /**
     * The list of [InboundCallPrice](https://www.twilio.com/docs/voice/pricing#inbound-call-price) records.
     */
    inboundCallPrices: Array<PricingV2TrunkingCountryInstanceOriginatingCallPrices>;
    /**
     * The currency in which prices are measured, specified in [ISO 4127](https://www.iso.org/iso/home/standards/currency_codes.htm) format (e.g. `usd`, `eur`, `jpy`).
     */
    priceUnit: string;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a CountryInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CountryInstance
     */
    fetch(callback?: (error: Error | null, item?: CountryInstance) => any): Promise<CountryInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        country: string;
        isoCountry: string;
        outboundPrefixPrices: PricingV2TrunkingCountryInstanceTerminatingPrefixPrices[];
        inboundCallPrices: PricingV2TrunkingCountryInstanceOriginatingCallPrices[];
        priceUnit: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CountrySolution {
}
export interface CountryListInstance {
    _version: V2;
    _solution: CountrySolution;
    _uri: string;
    (isoCountry: string): CountryContext;
    get(isoCountry: string): CountryContext;
    /**
     * Streams CountryInstance records from the API.
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
     * @param { CountryListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: CountryInstance, done: (err?: Error) => void) => void): void;
    each(params: CountryListInstanceEachOptions, callback?: (item: CountryInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of CountryInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: CountryPage) => any): Promise<CountryPage>;
    /**
     * Lists CountryInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CountryListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: CountryInstance[]) => any): Promise<CountryInstance[]>;
    list(params: CountryListInstanceOptions, callback?: (error: Error | null, items: CountryInstance[]) => any): Promise<CountryInstance[]>;
    /**
     * Retrieve a single page of CountryInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CountryListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: CountryPage) => any): Promise<CountryPage>;
    page(params: CountryListInstancePageOptions, callback?: (error: Error | null, items: CountryPage) => any): Promise<CountryPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CountryListInstance(version: V2): CountryListInstance;
export declare class CountryPage extends Page<V2, CountryPayload, CountryResource, CountryInstance> {
    /**
     * Initialize the CountryPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: CountrySolution);
    /**
     * Build an instance of CountryInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CountryResource): CountryInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
