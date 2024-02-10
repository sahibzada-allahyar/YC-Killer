/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2010 from "../../../V2010";
import { PhoneNumberCapabilities } from "../../../../../interfaces";
/**
 * Options to pass to each
 */
export interface SharedCostListInstanceEachOptions {
    /** The area code of the phone numbers to read. Applies to only phone numbers in the US and Canada. */
    areaCode?: number;
    /** The pattern on which to match phone numbers. Valid characters are `*`, `0-9`, `a-z`, and `A-Z`. The `*` character matches any single digit. For examples, see [Example 2](https://www.twilio.com/docs/phone-numbers/api/availablephonenumber-resource#local-get-basic-example-2) and [Example 3](https://www.twilio.com/docs/phone-numbers/api/availablephonenumber-resource#local-get-basic-example-3). If specified, this value must have at least two characters. */
    contains?: string;
    /** Whether the phone numbers can receive text messages. Can be: `true` or `false`. */
    smsEnabled?: boolean;
    /** Whether the phone numbers can receive MMS messages. Can be: `true` or `false`. */
    mmsEnabled?: boolean;
    /** Whether the phone numbers can receive calls. Can be: `true` or `false`. */
    voiceEnabled?: boolean;
    /** Whether to exclude phone numbers that require an [Address](https://www.twilio.com/docs/usage/api/address). Can be: `true` or `false` and the default is `false`. */
    excludeAllAddressRequired?: boolean;
    /** Whether to exclude phone numbers that require a local [Address](https://www.twilio.com/docs/usage/api/address). Can be: `true` or `false` and the default is `false`. */
    excludeLocalAddressRequired?: boolean;
    /** Whether to exclude phone numbers that require a foreign [Address](https://www.twilio.com/docs/usage/api/address). Can be: `true` or `false` and the default is `false`. */
    excludeForeignAddressRequired?: boolean;
    /** Whether to read phone numbers that are new to the Twilio platform. Can be: `true` or `false` and the default is `true`. */
    beta?: boolean;
    /** Given a phone number, find a geographically close number within `distance` miles. Distance defaults to 25 miles. Applies to only phone numbers in the US and Canada. */
    nearNumber?: string;
    /** Given a latitude/longitude pair `lat,long` find geographically close numbers within `distance` miles. Applies to only phone numbers in the US and Canada. */
    nearLatLong?: string;
    /** The search radius, in miles, for a `near_` query.  Can be up to `500` and the default is `25`. Applies to only phone numbers in the US and Canada. */
    distance?: number;
    /** Limit results to a particular postal code. Given a phone number, search within the same postal code as that number. Applies to only phone numbers in the US and Canada. */
    inPostalCode?: string;
    /** Limit results to a particular region, state, or province. Given a phone number, search within the same region as that number. Applies to only phone numbers in the US and Canada. */
    inRegion?: string;
    /** Limit results to a specific rate center, or given a phone number search within the same rate center as that number. Requires `in_lata` to be set as well. Applies to only phone numbers in the US and Canada. */
    inRateCenter?: string;
    /** Limit results to a specific local access and transport area ([LATA](https://en.wikipedia.org/wiki/Local_access_and_transport_area)). Given a phone number, search within the same [LATA](https://en.wikipedia.org/wiki/Local_access_and_transport_area) as that number. Applies to only phone numbers in the US and Canada. */
    inLata?: string;
    /** Limit results to a particular locality or city. Given a phone number, search within the same Locality as that number. */
    inLocality?: string;
    /** Whether the phone numbers can receive faxes. Can be: `true` or `false`. */
    faxEnabled?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SharedCostInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SharedCostListInstanceOptions {
    /** The area code of the phone numbers to read. Applies to only phone numbers in the US and Canada. */
    areaCode?: number;
    /** The pattern on which to match phone numbers. Valid characters are `*`, `0-9`, `a-z`, and `A-Z`. The `*` character matches any single digit. For examples, see [Example 2](https://www.twilio.com/docs/phone-numbers/api/availablephonenumber-resource#local-get-basic-example-2) and [Example 3](https://www.twilio.com/docs/phone-numbers/api/availablephonenumber-resource#local-get-basic-example-3). If specified, this value must have at least two characters. */
    contains?: string;
    /** Whether the phone numbers can receive text messages. Can be: `true` or `false`. */
    smsEnabled?: boolean;
    /** Whether the phone numbers can receive MMS messages. Can be: `true` or `false`. */
    mmsEnabled?: boolean;
    /** Whether the phone numbers can receive calls. Can be: `true` or `false`. */
    voiceEnabled?: boolean;
    /** Whether to exclude phone numbers that require an [Address](https://www.twilio.com/docs/usage/api/address). Can be: `true` or `false` and the default is `false`. */
    excludeAllAddressRequired?: boolean;
    /** Whether to exclude phone numbers that require a local [Address](https://www.twilio.com/docs/usage/api/address). Can be: `true` or `false` and the default is `false`. */
    excludeLocalAddressRequired?: boolean;
    /** Whether to exclude phone numbers that require a foreign [Address](https://www.twilio.com/docs/usage/api/address). Can be: `true` or `false` and the default is `false`. */
    excludeForeignAddressRequired?: boolean;
    /** Whether to read phone numbers that are new to the Twilio platform. Can be: `true` or `false` and the default is `true`. */
    beta?: boolean;
    /** Given a phone number, find a geographically close number within `distance` miles. Distance defaults to 25 miles. Applies to only phone numbers in the US and Canada. */
    nearNumber?: string;
    /** Given a latitude/longitude pair `lat,long` find geographically close numbers within `distance` miles. Applies to only phone numbers in the US and Canada. */
    nearLatLong?: string;
    /** The search radius, in miles, for a `near_` query.  Can be up to `500` and the default is `25`. Applies to only phone numbers in the US and Canada. */
    distance?: number;
    /** Limit results to a particular postal code. Given a phone number, search within the same postal code as that number. Applies to only phone numbers in the US and Canada. */
    inPostalCode?: string;
    /** Limit results to a particular region, state, or province. Given a phone number, search within the same region as that number. Applies to only phone numbers in the US and Canada. */
    inRegion?: string;
    /** Limit results to a specific rate center, or given a phone number search within the same rate center as that number. Requires `in_lata` to be set as well. Applies to only phone numbers in the US and Canada. */
    inRateCenter?: string;
    /** Limit results to a specific local access and transport area ([LATA](https://en.wikipedia.org/wiki/Local_access_and_transport_area)). Given a phone number, search within the same [LATA](https://en.wikipedia.org/wiki/Local_access_and_transport_area) as that number. Applies to only phone numbers in the US and Canada. */
    inLata?: string;
    /** Limit results to a particular locality or city. Given a phone number, search within the same Locality as that number. */
    inLocality?: string;
    /** Whether the phone numbers can receive faxes. Can be: `true` or `false`. */
    faxEnabled?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SharedCostListInstancePageOptions {
    /** The area code of the phone numbers to read. Applies to only phone numbers in the US and Canada. */
    areaCode?: number;
    /** The pattern on which to match phone numbers. Valid characters are `*`, `0-9`, `a-z`, and `A-Z`. The `*` character matches any single digit. For examples, see [Example 2](https://www.twilio.com/docs/phone-numbers/api/availablephonenumber-resource#local-get-basic-example-2) and [Example 3](https://www.twilio.com/docs/phone-numbers/api/availablephonenumber-resource#local-get-basic-example-3). If specified, this value must have at least two characters. */
    contains?: string;
    /** Whether the phone numbers can receive text messages. Can be: `true` or `false`. */
    smsEnabled?: boolean;
    /** Whether the phone numbers can receive MMS messages. Can be: `true` or `false`. */
    mmsEnabled?: boolean;
    /** Whether the phone numbers can receive calls. Can be: `true` or `false`. */
    voiceEnabled?: boolean;
    /** Whether to exclude phone numbers that require an [Address](https://www.twilio.com/docs/usage/api/address). Can be: `true` or `false` and the default is `false`. */
    excludeAllAddressRequired?: boolean;
    /** Whether to exclude phone numbers that require a local [Address](https://www.twilio.com/docs/usage/api/address). Can be: `true` or `false` and the default is `false`. */
    excludeLocalAddressRequired?: boolean;
    /** Whether to exclude phone numbers that require a foreign [Address](https://www.twilio.com/docs/usage/api/address). Can be: `true` or `false` and the default is `false`. */
    excludeForeignAddressRequired?: boolean;
    /** Whether to read phone numbers that are new to the Twilio platform. Can be: `true` or `false` and the default is `true`. */
    beta?: boolean;
    /** Given a phone number, find a geographically close number within `distance` miles. Distance defaults to 25 miles. Applies to only phone numbers in the US and Canada. */
    nearNumber?: string;
    /** Given a latitude/longitude pair `lat,long` find geographically close numbers within `distance` miles. Applies to only phone numbers in the US and Canada. */
    nearLatLong?: string;
    /** The search radius, in miles, for a `near_` query.  Can be up to `500` and the default is `25`. Applies to only phone numbers in the US and Canada. */
    distance?: number;
    /** Limit results to a particular postal code. Given a phone number, search within the same postal code as that number. Applies to only phone numbers in the US and Canada. */
    inPostalCode?: string;
    /** Limit results to a particular region, state, or province. Given a phone number, search within the same region as that number. Applies to only phone numbers in the US and Canada. */
    inRegion?: string;
    /** Limit results to a specific rate center, or given a phone number search within the same rate center as that number. Requires `in_lata` to be set as well. Applies to only phone numbers in the US and Canada. */
    inRateCenter?: string;
    /** Limit results to a specific local access and transport area ([LATA](https://en.wikipedia.org/wiki/Local_access_and_transport_area)). Given a phone number, search within the same [LATA](https://en.wikipedia.org/wiki/Local_access_and_transport_area) as that number. Applies to only phone numbers in the US and Canada. */
    inLata?: string;
    /** Limit results to a particular locality or city. Given a phone number, search within the same Locality as that number. */
    inLocality?: string;
    /** Whether the phone numbers can receive faxes. Can be: `true` or `false`. */
    faxEnabled?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SharedCostSolution {
    accountSid: string;
    countryCode: string;
}
export interface SharedCostListInstance {
    _version: V2010;
    _solution: SharedCostSolution;
    _uri: string;
    /**
     * Streams SharedCostInstance records from the API.
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
     * @param { SharedCostListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SharedCostInstance, done: (err?: Error) => void) => void): void;
    each(params: SharedCostListInstanceEachOptions, callback?: (item: SharedCostInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SharedCostInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SharedCostPage) => any): Promise<SharedCostPage>;
    /**
     * Lists SharedCostInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SharedCostListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SharedCostInstance[]) => any): Promise<SharedCostInstance[]>;
    list(params: SharedCostListInstanceOptions, callback?: (error: Error | null, items: SharedCostInstance[]) => any): Promise<SharedCostInstance[]>;
    /**
     * Retrieve a single page of SharedCostInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SharedCostListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SharedCostPage) => any): Promise<SharedCostPage>;
    page(params: SharedCostListInstancePageOptions, callback?: (error: Error | null, items: SharedCostPage) => any): Promise<SharedCostPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SharedCostListInstance(version: V2010, accountSid: string, countryCode: string): SharedCostListInstance;
interface SharedCostPayload extends TwilioResponsePayload {
    available_phone_numbers: SharedCostResource[];
}
interface SharedCostResource {
    friendly_name: string;
    phone_number: string;
    lata: string;
    locality: string;
    rate_center: string;
    latitude: number;
    longitude: number;
    region: string;
    postal_code: string;
    iso_country: string;
    address_requirements: string;
    beta: boolean;
    capabilities: PhoneNumberCapabilities;
}
export declare class SharedCostInstance {
    protected _version: V2010;
    constructor(_version: V2010, payload: SharedCostResource, accountSid: string, countryCode: string);
    /**
     * A formatted version of the phone number.
     */
    friendlyName: string;
    /**
     * The phone number in [E.164](https://www.twilio.com/docs/glossary/what-e164) format, which consists of a + followed by the country code and subscriber number.
     */
    phoneNumber: string;
    /**
     * The [LATA](https://en.wikipedia.org/wiki/Local_access_and_transport_area) of this phone number. Available for only phone numbers from the US and Canada.
     */
    lata: string;
    /**
     * The locality or city of this phone number\'s location.
     */
    locality: string;
    /**
     * The [rate center](https://en.wikipedia.org/wiki/Telephone_exchange) of this phone number. Available for only phone numbers from the US and Canada.
     */
    rateCenter: string;
    /**
     * The latitude of this phone number\'s location. Available for only phone numbers from the US and Canada.
     */
    latitude: number;
    /**
     * The longitude of this phone number\'s location. Available for only phone numbers from the US and Canada.
     */
    longitude: number;
    /**
     * The two-letter state or province abbreviation of this phone number\'s location. Available for only phone numbers from the US and Canada.
     */
    region: string;
    /**
     * The postal or ZIP code of this phone number\'s location. Available for only phone numbers from the US and Canada.
     */
    postalCode: string;
    /**
     * The [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of this phone number.
     */
    isoCountry: string;
    /**
     * The type of [Address](https://www.twilio.com/docs/usage/api/address) resource the phone number requires. Can be: `none`, `any`, `local`, or `foreign`. `none` means no address is required. `any` means an address is required, but it can be anywhere in the world. `local` means an address in the phone number\'s country is required. `foreign` means an address outside of the phone number\'s country is required.
     */
    addressRequirements: string;
    /**
     * Whether the phone number is new to the Twilio platform. Can be: `true` or `false`.
     */
    beta: boolean;
    capabilities: PhoneNumberCapabilities;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        friendlyName: string;
        phoneNumber: string;
        lata: string;
        locality: string;
        rateCenter: string;
        latitude: number;
        longitude: number;
        region: string;
        postalCode: string;
        isoCountry: string;
        addressRequirements: string;
        beta: boolean;
        capabilities: PhoneNumberCapabilities;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class SharedCostPage extends Page<V2010, SharedCostPayload, SharedCostResource, SharedCostInstance> {
    /**
     * Initialize the SharedCostPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: SharedCostSolution);
    /**
     * Build an instance of SharedCostInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SharedCostResource): SharedCostInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
