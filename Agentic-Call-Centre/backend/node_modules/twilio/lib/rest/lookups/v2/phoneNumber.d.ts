/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../V2";
export type PhoneNumberValidationError = "TOO_SHORT" | "TOO_LONG" | "INVALID_BUT_POSSIBLE" | "INVALID_COUNTRY_CODE" | "INVALID_LENGTH" | "NOT_A_NUMBER";
/**
 * Options to pass to fetch a PhoneNumberInstance
 */
export interface PhoneNumberContextFetchOptions {
    /** A comma-separated list of fields to return. Possible values are validation, caller_name, sim_swap, call_forwarding, line_status, line_type_intelligence, identity_match, reassigned_number, sms_pumping_risk, phone_number_quality_score. */
    fields?: string;
    /** The [country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) used if the phone number provided is in national format. */
    countryCode?: string;
    /** User’s first name. This query parameter is only used (optionally) for identity_match package requests. */
    firstName?: string;
    /** User’s last name. This query parameter is only used (optionally) for identity_match package requests. */
    lastName?: string;
    /** User’s first address line. This query parameter is only used (optionally) for identity_match package requests. */
    addressLine1?: string;
    /** User’s second address line. This query parameter is only used (optionally) for identity_match package requests. */
    addressLine2?: string;
    /** User’s city. This query parameter is only used (optionally) for identity_match package requests. */
    city?: string;
    /** User’s country subdivision, such as state, province, or locality. This query parameter is only used (optionally) for identity_match package requests. */
    state?: string;
    /** User’s postal zip code. This query parameter is only used (optionally) for identity_match package requests. */
    postalCode?: string;
    /** User’s country, up to two characters. This query parameter is only used (optionally) for identity_match package requests. */
    addressCountryCode?: string;
    /** User’s national ID, such as SSN or Passport ID. This query parameter is only used (optionally) for identity_match package requests. */
    nationalId?: string;
    /** User’s date of birth, in YYYYMMDD format. This query parameter is only used (optionally) for identity_match package requests. */
    dateOfBirth?: string;
    /** The date you obtained consent to call or text the end-user of the phone number or a date on which you are reasonably certain that the end-user could still be reached at that number. This query parameter is only used (optionally) for reassigned_number package requests. */
    lastVerifiedDate?: string;
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
    protected _version: V2;
    protected _solution: PhoneNumberContextSolution;
    protected _uri: string;
    constructor(_version: V2, phoneNumber: string);
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
    calling_country_code: string;
    country_code: string;
    phone_number: string;
    national_format: string;
    valid: boolean;
    validation_errors: Array<PhoneNumberValidationError>;
    caller_name: any;
    sim_swap: any;
    call_forwarding: any;
    line_status: any;
    line_type_intelligence: any;
    identity_match: any;
    reassigned_number: any;
    sms_pumping_risk: any;
    phone_number_quality_score: any;
    url: string;
}
export declare class PhoneNumberInstance {
    protected _version: V2;
    protected _solution: PhoneNumberContextSolution;
    protected _context?: PhoneNumberContext;
    constructor(_version: V2, payload: PhoneNumberResource, phoneNumber?: string);
    /**
     * International dialing prefix of the phone number defined in the E.164 standard.
     */
    callingCountryCode: string;
    /**
     * The phone number\'s [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2).
     */
    countryCode: string;
    /**
     * The phone number in [E.164](https://www.twilio.com/docs/glossary/what-e164) format, which consists of a + followed by the country code and subscriber number.
     */
    phoneNumber: string;
    /**
     * The phone number in [national format](https://en.wikipedia.org/wiki/National_conventions_for_writing_telephone_numbers).
     */
    nationalFormat: string;
    /**
     * Boolean which indicates if the phone number is in a valid range that can be freely assigned by a carrier to a user.
     */
    valid: boolean;
    /**
     * Contains reasons why a phone number is invalid. Possible values: TOO_SHORT, TOO_LONG, INVALID_BUT_POSSIBLE, INVALID_COUNTRY_CODE, INVALID_LENGTH, NOT_A_NUMBER.
     */
    validationErrors: Array<PhoneNumberValidationError>;
    /**
     * An object that contains caller name information based on [CNAM](https://support.twilio.com/hc/en-us/articles/360051670533-Getting-Started-with-CNAM-Caller-ID).
     */
    callerName: any;
    /**
     * An object that contains information on the last date the subscriber identity module (SIM) was changed for a mobile phone number.
     */
    simSwap: any;
    /**
     * An object that contains information on the unconditional call forwarding status of mobile phone number.
     */
    callForwarding: any;
    /**
     * An object that contains line status information for a mobile phone number.
     */
    lineStatus: any;
    /**
     * An object that contains line type information including the carrier name, mobile country code, and mobile network code.
     */
    lineTypeIntelligence: any;
    /**
     * An object that contains identity match information. The result of comparing user-provided information including name, address, date of birth, national ID, against authoritative phone-based data sources
     */
    identityMatch: any;
    /**
     * An object that contains reassigned number information. Reassigned Numbers will return a phone number\'s reassignment status given a phone number and date
     */
    reassignedNumber: any;
    /**
     * An object that contains information on if a phone number has been currently or previously blocked by Verify Fraud Guard for receiving malicious SMS pumping traffic as well as other signals associated with risky carriers and low conversion rates.
     */
    smsPumpingRisk: any;
    /**
     * An object that contains information of a mobile phone number quality score. Quality score will return a risk score about the phone number.
     */
    phoneNumberQualityScore: any;
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
        callingCountryCode: string;
        countryCode: string;
        phoneNumber: string;
        nationalFormat: string;
        valid: boolean;
        validationErrors: PhoneNumberValidationError[];
        callerName: any;
        simSwap: any;
        callForwarding: any;
        lineStatus: any;
        lineTypeIntelligence: any;
        identityMatch: any;
        reassignedNumber: any;
        smsPumpingRisk: any;
        phoneNumberQualityScore: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PhoneNumberSolution {
}
export interface PhoneNumberListInstance {
    _version: V2;
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
export declare function PhoneNumberListInstance(version: V2): PhoneNumberListInstance;
export {};
