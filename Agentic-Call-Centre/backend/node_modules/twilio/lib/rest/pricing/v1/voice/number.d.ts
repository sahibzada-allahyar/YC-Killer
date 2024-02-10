/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
/**
 * The [InboundCallPrice](https://www.twilio.com/docs/voice/pricing#inbound-call-price) record. If `null`, the Phone Number is not a Twilio number owned by this account.
 */
export declare class PricingV1VoiceVoiceNumberInboundCallPrice {
    "basePrice"?: number;
    "currentPrice"?: number;
    "numberType"?: string;
}
/**
 * The OutboundCallPrice record, which includes a list of `origination_prefixes` and the `base_price` and `current_price` for those prefixes.
 */
export declare class PricingV1VoiceVoiceNumberOutboundCallPrice {
    "basePrice"?: number;
    "currentPrice"?: number;
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
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface NumberContextSolution {
    number: string;
}
export declare class NumberContextImpl implements NumberContext {
    protected _version: V1;
    protected _solution: NumberContextSolution;
    protected _uri: string;
    constructor(_version: V1, number: string);
    fetch(callback?: (error: Error | null, item?: NumberInstance) => any): Promise<NumberInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): NumberContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface NumberResource {
    number: string;
    country: string;
    iso_country: string;
    outbound_call_price: PricingV1VoiceVoiceNumberOutboundCallPrice;
    inbound_call_price: PricingV1VoiceVoiceNumberInboundCallPrice;
    price_unit: string;
    url: string;
}
export declare class NumberInstance {
    protected _version: V1;
    protected _solution: NumberContextSolution;
    protected _context?: NumberContext;
    constructor(_version: V1, payload: NumberResource, number?: string);
    /**
     * The phone number.
     */
    number: string;
    /**
     * The name of the country.
     */
    country: string;
    /**
     * The [ISO country code](http://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
     */
    isoCountry: string;
    outboundCallPrice: PricingV1VoiceVoiceNumberOutboundCallPrice;
    inboundCallPrice: PricingV1VoiceVoiceNumberInboundCallPrice;
    /**
     * The currency in which prices are measured, specified in [ISO 4127](http://www.iso.org/iso/home/standards/currency_codes.htm) format (e.g. `usd`, `eur`, `jpy`).
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
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        number: string;
        country: string;
        isoCountry: string;
        outboundCallPrice: PricingV1VoiceVoiceNumberOutboundCallPrice;
        inboundCallPrice: PricingV1VoiceVoiceNumberInboundCallPrice;
        priceUnit: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface NumberSolution {
}
export interface NumberListInstance {
    _version: V1;
    _solution: NumberSolution;
    _uri: string;
    (number: string): NumberContext;
    get(number: string): NumberContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function NumberListInstance(version: V1): NumberListInstance;
export {};
