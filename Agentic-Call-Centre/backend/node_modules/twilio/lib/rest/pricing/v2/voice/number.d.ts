/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../../V2";
/**
 * The [InboundCallPrice](https://www.twilio.com/docs/voice/pricing#inbound-call-price) record.
 */
export declare class PricingV2VoiceVoiceNumberInboundCallPrice {
    "basePrice"?: number;
    "currentPrice"?: number;
    "numberType"?: string;
}
export declare class PricingV2VoiceVoiceNumberOutboundCallPrices {
    "basePrice"?: number;
    "currentPrice"?: number;
    "originationPrefixes"?: Array<string>;
}
/**
 * Options to pass to fetch a NumberInstance
 */
export interface NumberContextFetchOptions {
    /** The origination phone number, in [E.164](https://www.twilio.com/docs/glossary/what-e164) format, for which to fetch the origin-based voice pricing information. E.164 format consists of a + followed by the country code and subscriber number. */
    originationNumber?: string;
}
export interface NumberContext {
    /**
     * Fetch a NumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NumberInstance
     */
    fetch(callback?: (error: Error | null, item?: NumberInstance) => any): Promise<NumberInstance>;
    /**
     * Fetch a NumberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NumberInstance
     */
    fetch(params: NumberContextFetchOptions, callback?: (error: Error | null, item?: NumberInstance) => any): Promise<NumberInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface NumberContextSolution {
    destinationNumber: string;
}
export declare class NumberContextImpl implements NumberContext {
    protected _version: V2;
    protected _solution: NumberContextSolution;
    protected _uri: string;
    constructor(_version: V2, destinationNumber: string);
    fetch(params?: NumberContextFetchOptions | ((error: Error | null, item?: NumberInstance) => any), callback?: (error: Error | null, item?: NumberInstance) => any): Promise<NumberInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): NumberContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface NumberResource {
    destination_number: string;
    origination_number: string;
    country: string;
    iso_country: string;
    outbound_call_prices: Array<PricingV2VoiceVoiceNumberOutboundCallPrices>;
    inbound_call_price: PricingV2VoiceVoiceNumberInboundCallPrice;
    price_unit: string;
    url: string;
}
export declare class NumberInstance {
    protected _version: V2;
    protected _solution: NumberContextSolution;
    protected _context?: NumberContext;
    constructor(_version: V2, payload: NumberResource, destinationNumber?: string);
    /**
     * The destination phone number in [E.164](https://www.twilio.com/docs/glossary/what-e164) format, which consists of a + followed by the country code and subscriber number.
     */
    destinationNumber: string;
    /**
     * The origination phone number in [[E.164](https://www.twilio.com/docs/glossary/what-e164) format, which consists of a + followed by the country code and subscriber number.
     */
    originationNumber: string;
    /**
     * The name of the country.
     */
    country: string;
    /**
     * The [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2)
     */
    isoCountry: string;
    /**
     * The list of [OutboundCallPriceWithOrigin](https://www.twilio.com/docs/voice/pricing#outbound-call-price-with-origin) records.
     */
    outboundCallPrices: Array<PricingV2VoiceVoiceNumberOutboundCallPrices>;
    inboundCallPrice: PricingV2VoiceVoiceNumberInboundCallPrice;
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
     * Fetch a NumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NumberInstance
     */
    fetch(callback?: (error: Error | null, item?: NumberInstance) => any): Promise<NumberInstance>;
    /**
     * Fetch a NumberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NumberInstance
     */
    fetch(params: NumberContextFetchOptions, callback?: (error: Error | null, item?: NumberInstance) => any): Promise<NumberInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        destinationNumber: string;
        originationNumber: string;
        country: string;
        isoCountry: string;
        outboundCallPrices: PricingV2VoiceVoiceNumberOutboundCallPrices[];
        inboundCallPrice: PricingV2VoiceVoiceNumberInboundCallPrice;
        priceUnit: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface NumberSolution {
}
export interface NumberListInstance {
    _version: V2;
    _solution: NumberSolution;
    _uri: string;
    (destinationNumber: string): NumberContext;
    get(destinationNumber: string): NumberContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function NumberListInstance(version: V2): NumberListInstance;
export {};
