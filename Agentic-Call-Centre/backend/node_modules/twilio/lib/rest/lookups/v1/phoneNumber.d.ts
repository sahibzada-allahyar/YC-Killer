/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to fetch a PhoneNumberInstance
 */
export interface PhoneNumberContextFetchOptions {
    /** The [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the phone number to fetch. This is used to specify the country when the phone number is provided in a national format. */
    countryCode?: string;
    /** The type of information to return. Can be: `carrier` or `caller-name`. The default is null.  Carrier information costs $0.005 per phone number looked up.  Caller Name information is currently available only in the US and costs $0.01 per phone number looked up.  To retrieve both types on information, specify this parameter twice; once with `carrier` and once with `caller-name` as the value. */
    type?: Array<string>;
    /** The `unique_name` of an Add-on you would like to invoke. Can be the `unique_name` of an Add-on that is installed on your account. You can specify multiple instances of this parameter to invoke multiple Add-ons. For more information about  Add-ons, see the [Add-ons documentation](https://www.twilio.com/docs/add-ons). */
    addOns?: Array<string>;
    /** Data specific to the add-on you would like to invoke. The content and format of this value depends on the add-on. */
    addOnsData?: Record<string, object>;
}
export interface PhoneNumberContext {
    /**
     * Fetch a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    fetch(callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Fetch a PhoneNumberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    fetch(params: PhoneNumberContextFetchOptions, callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PhoneNumberContextSolution {
    phoneNumber: string;
}
export declare class PhoneNumberContextImpl implements PhoneNumberContext {
    protected _version: V1;
    protected _solution: PhoneNumberContextSolution;
    protected _uri: string;
    constructor(_version: V1, phoneNumber: string);
    fetch(params?: PhoneNumberContextFetchOptions | ((error: Error | null, item?: PhoneNumberInstance) => any), callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PhoneNumberContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PhoneNumberResource {
    caller_name: any;
    country_code: string;
    phone_number: string;
    national_format: string;
    carrier: any;
    add_ons: any;
    url: string;
}
export declare class PhoneNumberInstance {
    protected _version: V1;
    protected _solution: PhoneNumberContextSolution;
    protected _context?: PhoneNumberContext;
    constructor(_version: V1, payload: PhoneNumberResource, phoneNumber?: string);
    /**
     * The name of the phone number\'s owner. If `null`, that information was not available.
     */
    callerName: any;
    /**
     * The [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) for the phone number.
     */
    countryCode: string;
    /**
     * The phone number in [E.164](https://www.twilio.com/docs/glossary/what-e164) format, which consists of a + followed by the country code and subscriber number.
     */
    phoneNumber: string;
    /**
     * The phone number, in national format.
     */
    nationalFormat: string;
    /**
     * The telecom company that provides the phone number.
     */
    carrier: any;
    /**
     * A JSON string with the results of the Add-ons you specified in the `add_ons` parameters. For the format of the object, see [Using Add-ons](https://www.twilio.com/docs/add-ons).
     */
    addOns: any;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    fetch(callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Fetch a PhoneNumberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    fetch(params: PhoneNumberContextFetchOptions, callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        callerName: any;
        countryCode: string;
        phoneNumber: string;
        nationalFormat: string;
        carrier: any;
        addOns: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PhoneNumberSolution {
}
export interface PhoneNumberListInstance {
    _version: V1;
    _solution: PhoneNumberSolution;
    _uri: string;
    (phoneNumber: string): PhoneNumberContext;
    get(phoneNumber: string): PhoneNumberContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PhoneNumberListInstance(version: V1): PhoneNumberListInstance;
export {};
