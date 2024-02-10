/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { HighriskSpecialPrefixListInstance } from "./country/highriskSpecialPrefix";
/**
 * Options to pass to each
 */
export interface CountryListInstanceEachOptions {
    /** Filter to retrieve the country permissions by specifying the [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) */
    isoCode?: string;
    /** Filter to retrieve the country permissions by specifying the continent */
    continent?: string;
    /** Filter the results by specified [country codes](https://www.itu.int/itudoc/itu-t/ob-lists/icc/e164_763.html) */
    countryCode?: string;
    /** Filter to retrieve the country permissions with dialing to low-risk numbers enabled. Can be: `true` or `false`. */
    lowRiskNumbersEnabled?: boolean;
    /** Filter to retrieve the country permissions with dialing to high-risk special service numbers enabled. Can be: `true` or `false` */
    highRiskSpecialNumbersEnabled?: boolean;
    /** Filter to retrieve the country permissions with dialing to high-risk [toll fraud](https://www.twilio.com/blog/how-to-protect-your-account-from-toll-fraud-with-voice-dialing-geo-permissions-html) numbers enabled. Can be: `true` or `false`. */
    highRiskTollfraudNumbersEnabled?: boolean;
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
    /** Filter to retrieve the country permissions by specifying the [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) */
    isoCode?: string;
    /** Filter to retrieve the country permissions by specifying the continent */
    continent?: string;
    /** Filter the results by specified [country codes](https://www.itu.int/itudoc/itu-t/ob-lists/icc/e164_763.html) */
    countryCode?: string;
    /** Filter to retrieve the country permissions with dialing to low-risk numbers enabled. Can be: `true` or `false`. */
    lowRiskNumbersEnabled?: boolean;
    /** Filter to retrieve the country permissions with dialing to high-risk special service numbers enabled. Can be: `true` or `false` */
    highRiskSpecialNumbersEnabled?: boolean;
    /** Filter to retrieve the country permissions with dialing to high-risk [toll fraud](https://www.twilio.com/blog/how-to-protect-your-account-from-toll-fraud-with-voice-dialing-geo-permissions-html) numbers enabled. Can be: `true` or `false`. */
    highRiskTollfraudNumbersEnabled?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CountryListInstancePageOptions {
    /** Filter to retrieve the country permissions by specifying the [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) */
    isoCode?: string;
    /** Filter to retrieve the country permissions by specifying the continent */
    continent?: string;
    /** Filter the results by specified [country codes](https://www.itu.int/itudoc/itu-t/ob-lists/icc/e164_763.html) */
    countryCode?: string;
    /** Filter to retrieve the country permissions with dialing to low-risk numbers enabled. Can be: `true` or `false`. */
    lowRiskNumbersEnabled?: boolean;
    /** Filter to retrieve the country permissions with dialing to high-risk special service numbers enabled. Can be: `true` or `false` */
    highRiskSpecialNumbersEnabled?: boolean;
    /** Filter to retrieve the country permissions with dialing to high-risk [toll fraud](https://www.twilio.com/blog/how-to-protect-your-account-from-toll-fraud-with-voice-dialing-geo-permissions-html) numbers enabled. Can be: `true` or `false`. */
    highRiskTollfraudNumbersEnabled?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CountryContext {
    highriskSpecialPrefixes: HighriskSpecialPrefixListInstance;
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
    isoCode: string;
}
export declare class CountryContextImpl implements CountryContext {
    protected _version: V1;
    protected _solution: CountryContextSolution;
    protected _uri: string;
    protected _highriskSpecialPrefixes?: HighriskSpecialPrefixListInstance;
    constructor(_version: V1, isoCode: string);
    get highriskSpecialPrefixes(): HighriskSpecialPrefixListInstance;
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
    content: CountryResource[];
}
interface CountryResource {
    iso_code: string;
    name: string;
    continent: string;
    country_codes: Array<string>;
    low_risk_numbers_enabled: boolean;
    high_risk_special_numbers_enabled: boolean;
    high_risk_tollfraud_numbers_enabled: boolean;
    url: string;
    links: Record<string, string>;
}
export declare class CountryInstance {
    protected _version: V1;
    protected _solution: CountryContextSolution;
    protected _context?: CountryContext;
    constructor(_version: V1, payload: CountryResource, isoCode?: string);
    /**
     * The [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
     */
    isoCode: string;
    /**
     * The name of the country.
     */
    name: string;
    /**
     * The name of the continent in which the country is located.
     */
    continent: string;
    /**
     * The E.164 assigned [country codes(s)](https://www.itu.int/itudoc/itu-t/ob-lists/icc/e164_763.html)
     */
    countryCodes: Array<string>;
    /**
     * Whether dialing to low-risk numbers is enabled.
     */
    lowRiskNumbersEnabled: boolean;
    /**
     * Whether dialing to high-risk special services numbers is enabled. These prefixes include number ranges allocated by the country and include premium numbers, special services, shared cost, and others
     */
    highRiskSpecialNumbersEnabled: boolean;
    /**
     * Whether dialing to high-risk [toll fraud](https://www.twilio.com/blog/how-to-protect-your-account-from-toll-fraud-with-voice-dialing-geo-permissions-html) numbers is enabled. These prefixes include narrow number ranges that have a high-risk of international revenue sharing fraud (IRSF) attacks, also known as [toll fraud](https://www.twilio.com/blog/how-to-protect-your-account-from-toll-fraud-with-voice-dialing-geo-permissions-html). These prefixes are collected from anti-fraud databases and verified by analyzing calls on our network. These prefixes are not available for download and are updated frequently
     */
    highRiskTollfraudNumbersEnabled: boolean;
    /**
     * The absolute URL of this resource.
     */
    url: string;
    /**
     * A list of URLs related to this resource.
     */
    links: Record<string, string>;
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
     * Access the highriskSpecialPrefixes.
     */
    highriskSpecialPrefixes(): HighriskSpecialPrefixListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        isoCode: string;
        name: string;
        continent: string;
        countryCodes: string[];
        lowRiskNumbersEnabled: boolean;
        highRiskSpecialNumbersEnabled: boolean;
        highRiskTollfraudNumbersEnabled: boolean;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CountrySolution {
}
export interface CountryListInstance {
    _version: V1;
    _solution: CountrySolution;
    _uri: string;
    (isoCode: string): CountryContext;
    get(isoCode: string): CountryContext;
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
export declare function CountryListInstance(version: V1): CountryListInstance;
export declare class CountryPage extends Page<V1, CountryPayload, CountryResource, CountryInstance> {
    /**
     * Initialize the CountryPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: CountrySolution);
    /**
     * Build an instance of CountryInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CountryResource): CountryInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
