/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2010 from "../../V2010";
import { DependentPhoneNumberListInstance } from "./address/dependentPhoneNumber";
/**
 * Options to pass to update a AddressInstance
 */
export interface AddressContextUpdateOptions {
    /** A descriptive string that you create to describe the address. It can be up to 64 characters long. */
    friendlyName?: string;
    /** The name to associate with the address. */
    customerName?: string;
    /** The number and street address of the address. */
    street?: string;
    /** The city of the address. */
    city?: string;
    /** The state or region of the address. */
    region?: string;
    /** The postal code of the address. */
    postalCode?: string;
    /** Whether to enable emergency calling on the address. Can be: `true` or `false`. */
    emergencyEnabled?: boolean;
    /** Whether we should automatically correct the address. Can be: `true` or `false` and the default is `true`. If empty or `true`, we will correct the address you provide if necessary. If `false`, we won\\\'t alter the address you provide. */
    autoCorrectAddress?: boolean;
    /** The additional number and street address of the address. */
    streetSecondary?: string;
}
/**
 * Options to pass to create a AddressInstance
 */
export interface AddressListInstanceCreateOptions {
    /** The name to associate with the new address. */
    customerName: string;
    /** The number and street address of the new address. */
    street: string;
    /** The city of the new address. */
    city: string;
    /** The state or region of the new address. */
    region: string;
    /** The postal code of the new address. */
    postalCode: string;
    /** The ISO country code of the new address. */
    isoCountry: string;
    /** A descriptive string that you create to describe the new address. It can be up to 64 characters long. */
    friendlyName?: string;
    /** Whether to enable emergency calling on the new address. Can be: `true` or `false`. */
    emergencyEnabled?: boolean;
    /** Whether we should automatically correct the address. Can be: `true` or `false` and the default is `true`. If empty or `true`, we will correct the address you provide if necessary. If `false`, we won\\\'t alter the address you provide. */
    autoCorrectAddress?: boolean;
    /** The additional number and street address of the address. */
    streetSecondary?: string;
}
/**
 * Options to pass to each
 */
export interface AddressListInstanceEachOptions {
    /** The `customer_name` of the Address resources to read. */
    customerName?: string;
    /** The string that identifies the Address resources to read. */
    friendlyName?: string;
    /** The ISO country code of the Address resources to read. */
    isoCountry?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AddressInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AddressListInstanceOptions {
    /** The `customer_name` of the Address resources to read. */
    customerName?: string;
    /** The string that identifies the Address resources to read. */
    friendlyName?: string;
    /** The ISO country code of the Address resources to read. */
    isoCountry?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AddressListInstancePageOptions {
    /** The `customer_name` of the Address resources to read. */
    customerName?: string;
    /** The string that identifies the Address resources to read. */
    friendlyName?: string;
    /** The ISO country code of the Address resources to read. */
    isoCountry?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AddressContext {
    dependentPhoneNumbers: DependentPhoneNumberListInstance;
    /**
     * Remove a AddressInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AddressInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AddressInstance
     */
    fetch(callback?: (error: Error | null, item?: AddressInstance) => any): Promise<AddressInstance>;
    /**
     * Update a AddressInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AddressInstance
     */
    update(callback?: (error: Error | null, item?: AddressInstance) => any): Promise<AddressInstance>;
    /**
     * Update a AddressInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AddressInstance
     */
    update(params: AddressContextUpdateOptions, callback?: (error: Error | null, item?: AddressInstance) => any): Promise<AddressInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AddressContextSolution {
    accountSid: string;
    sid: string;
}
export declare class AddressContextImpl implements AddressContext {
    protected _version: V2010;
    protected _solution: AddressContextSolution;
    protected _uri: string;
    protected _dependentPhoneNumbers?: DependentPhoneNumberListInstance;
    constructor(_version: V2010, accountSid: string, sid: string);
    get dependentPhoneNumbers(): DependentPhoneNumberListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AddressInstance) => any): Promise<AddressInstance>;
    update(params?: AddressContextUpdateOptions | ((error: Error | null, item?: AddressInstance) => any), callback?: (error: Error | null, item?: AddressInstance) => any): Promise<AddressInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AddressContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AddressPayload extends TwilioResponsePayload {
    addresses: AddressResource[];
}
interface AddressResource {
    account_sid: string;
    city: string;
    customer_name: string;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    iso_country: string;
    postal_code: string;
    region: string;
    sid: string;
    street: string;
    uri: string;
    emergency_enabled: boolean;
    validated: boolean;
    verified: boolean;
    street_secondary: string;
}
export declare class AddressInstance {
    protected _version: V2010;
    protected _solution: AddressContextSolution;
    protected _context?: AddressContext;
    constructor(_version: V2010, payload: AddressResource, accountSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that is responsible for the Address resource.
     */
    accountSid: string;
    /**
     * The city in which the address is located.
     */
    city: string;
    /**
     * The name associated with the address.This property has a maximum length of 16 4-byte characters, or 21 3-byte characters.
     */
    customerName: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The ISO country code of the address.
     */
    isoCountry: string;
    /**
     * The postal code of the address.
     */
    postalCode: string;
    /**
     * The state or region of the address.
     */
    region: string;
    /**
     * The unique string that that we created to identify the Address resource.
     */
    sid: string;
    /**
     * The number and street address of the address.
     */
    street: string;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    /**
     * Whether emergency calling has been enabled on this number.
     */
    emergencyEnabled: boolean;
    /**
     * Whether the address has been validated to comply with local regulation. In countries that require valid addresses, an invalid address will not be accepted. `true` indicates the Address has been validated. `false` indicate the country doesn\'t require validation or the Address is not valid.
     */
    validated: boolean;
    /**
     * Whether the address has been verified to comply with regulation. In countries that require valid addresses, an invalid address will not be accepted. `true` indicates the Address has been verified. `false` indicate the country doesn\'t require verified or the Address is not valid.
     */
    verified: boolean;
    /**
     * The additional number and street address of the address.
     */
    streetSecondary: string;
    private get _proxy();
    /**
     * Remove a AddressInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AddressInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AddressInstance
     */
    fetch(callback?: (error: Error | null, item?: AddressInstance) => any): Promise<AddressInstance>;
    /**
     * Update a AddressInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AddressInstance
     */
    update(callback?: (error: Error | null, item?: AddressInstance) => any): Promise<AddressInstance>;
    /**
     * Update a AddressInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AddressInstance
     */
    update(params: AddressContextUpdateOptions, callback?: (error: Error | null, item?: AddressInstance) => any): Promise<AddressInstance>;
    /**
     * Access the dependentPhoneNumbers.
     */
    dependentPhoneNumbers(): DependentPhoneNumberListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        city: string;
        customerName: string;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        isoCountry: string;
        postalCode: string;
        region: string;
        sid: string;
        street: string;
        uri: string;
        emergencyEnabled: boolean;
        validated: boolean;
        verified: boolean;
        streetSecondary: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AddressSolution {
    accountSid: string;
}
export interface AddressListInstance {
    _version: V2010;
    _solution: AddressSolution;
    _uri: string;
    (sid: string): AddressContext;
    get(sid: string): AddressContext;
    /**
     * Create a AddressInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AddressInstance
     */
    create(params: AddressListInstanceCreateOptions, callback?: (error: Error | null, item?: AddressInstance) => any): Promise<AddressInstance>;
    /**
     * Streams AddressInstance records from the API.
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
     * @param { AddressListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AddressInstance, done: (err?: Error) => void) => void): void;
    each(params: AddressListInstanceEachOptions, callback?: (item: AddressInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AddressInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AddressPage) => any): Promise<AddressPage>;
    /**
     * Lists AddressInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AddressListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AddressInstance[]) => any): Promise<AddressInstance[]>;
    list(params: AddressListInstanceOptions, callback?: (error: Error | null, items: AddressInstance[]) => any): Promise<AddressInstance[]>;
    /**
     * Retrieve a single page of AddressInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AddressListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AddressPage) => any): Promise<AddressPage>;
    page(params: AddressListInstancePageOptions, callback?: (error: Error | null, items: AddressPage) => any): Promise<AddressPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AddressListInstance(version: V2010, accountSid: string): AddressListInstance;
export declare class AddressPage extends Page<V2010, AddressPayload, AddressResource, AddressInstance> {
    /**
     * Initialize the AddressPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: AddressSolution);
    /**
     * Build an instance of AddressInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AddressResource): AddressInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
