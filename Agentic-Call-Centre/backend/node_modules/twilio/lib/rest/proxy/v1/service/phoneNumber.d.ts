/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { PhoneNumberCapabilities } from "../../../../interfaces";
/**
 * Options to pass to update a PhoneNumberInstance
 */
export interface PhoneNumberContextUpdateOptions {
    /** Whether the phone number should be reserved and not be assigned to a participant using proxy pool logic. See [Reserved Phone Numbers](https://www.twilio.com/docs/proxy/reserved-phone-numbers) for more information. */
    isReserved?: boolean;
}
/**
 * Options to pass to create a PhoneNumberInstance
 */
export interface PhoneNumberListInstanceCreateOptions {
    /** The SID of a Twilio [IncomingPhoneNumber](https://www.twilio.com/docs/phone-numbers/api/incomingphonenumber-resource) resource that represents the Twilio Number you would like to assign to your Proxy Service. */
    sid?: string;
    /** The phone number in [E.164](https://www.twilio.com/docs/glossary/what-e164) format.  E.164 phone numbers consist of a + followed by the country code and subscriber number without punctuation characters. For example, +14155551234. */
    phoneNumber?: string;
    /** Whether the new phone number should be reserved and not be assigned to a participant using proxy pool logic. See [Reserved Phone Numbers](https://www.twilio.com/docs/proxy/reserved-phone-numbers) for more information. */
    isReserved?: boolean;
}
/**
 * Options to pass to each
 */
export interface PhoneNumberListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: PhoneNumberInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface PhoneNumberListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface PhoneNumberListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface PhoneNumberContext {
    /**
     * Remove a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    fetch(callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Update a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    update(callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Update a PhoneNumberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    update(params: PhoneNumberContextUpdateOptions, callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PhoneNumberContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class PhoneNumberContextImpl implements PhoneNumberContext {
    protected _version: V1;
    protected _solution: PhoneNumberContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    update(params?: PhoneNumberContextUpdateOptions | ((error: Error | null, item?: PhoneNumberInstance) => any), callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PhoneNumberContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PhoneNumberPayload extends TwilioResponsePayload {
    phone_numbers: PhoneNumberResource[];
}
interface PhoneNumberResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    date_created: Date;
    date_updated: Date;
    phone_number: string;
    friendly_name: string;
    iso_country: string;
    capabilities: PhoneNumberCapabilities;
    url: string;
    is_reserved: boolean;
    in_use: number;
}
export declare class PhoneNumberInstance {
    protected _version: V1;
    protected _solution: PhoneNumberContextSolution;
    protected _context?: PhoneNumberContext;
    constructor(_version: V1, payload: PhoneNumberResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the PhoneNumber resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the PhoneNumber resource.
     */
    accountSid: string;
    /**
     * The SID of the PhoneNumber resource\'s parent [Service](https://www.twilio.com/docs/proxy/api/service) resource.
     */
    serviceSid: string;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the resource was created.
     */
    dateCreated: Date;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the resource was last updated.
     */
    dateUpdated: Date;
    /**
     * The phone number in [E.164](https://www.twilio.com/docs/glossary/what-e164) format, which consists of a + followed by the country code and subscriber number.
     */
    phoneNumber: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The ISO Country Code for the phone number.
     */
    isoCountry: string;
    capabilities: PhoneNumberCapabilities;
    /**
     * The absolute URL of the PhoneNumber resource.
     */
    url: string;
    /**
     * Whether the phone number should be reserved and not be assigned to a participant using proxy pool logic. See [Reserved Phone Numbers](https://www.twilio.com/docs/proxy/reserved-phone-numbers) for more information.
     */
    isReserved: boolean;
    /**
     * The number of open session assigned to the number. See the [How many Phone Numbers do I need?](https://www.twilio.com/docs/proxy/phone-numbers-needed) guide for more information.
     */
    inUse: number;
    private get _proxy();
    /**
     * Remove a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    fetch(callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Update a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    update(callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Update a PhoneNumberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    update(params: PhoneNumberContextUpdateOptions, callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        phoneNumber: string;
        friendlyName: string;
        isoCountry: string;
        capabilities: PhoneNumberCapabilities;
        url: string;
        isReserved: boolean;
        inUse: number;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PhoneNumberSolution {
    serviceSid: string;
}
export interface PhoneNumberListInstance {
    _version: V1;
    _solution: PhoneNumberSolution;
    _uri: string;
    (sid: string): PhoneNumberContext;
    get(sid: string): PhoneNumberContext;
    /**
     * Create a PhoneNumberInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    create(callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Create a PhoneNumberInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PhoneNumberInstance
     */
    create(params: PhoneNumberListInstanceCreateOptions, callback?: (error: Error | null, item?: PhoneNumberInstance) => any): Promise<PhoneNumberInstance>;
    /**
     * Streams PhoneNumberInstance records from the API.
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
     * @param { PhoneNumberListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: PhoneNumberInstance, done: (err?: Error) => void) => void): void;
    each(params: PhoneNumberListInstanceEachOptions, callback?: (item: PhoneNumberInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of PhoneNumberInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: PhoneNumberPage) => any): Promise<PhoneNumberPage>;
    /**
     * Lists PhoneNumberInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PhoneNumberListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: PhoneNumberInstance[]) => any): Promise<PhoneNumberInstance[]>;
    list(params: PhoneNumberListInstanceOptions, callback?: (error: Error | null, items: PhoneNumberInstance[]) => any): Promise<PhoneNumberInstance[]>;
    /**
     * Retrieve a single page of PhoneNumberInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { PhoneNumberListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: PhoneNumberPage) => any): Promise<PhoneNumberPage>;
    page(params: PhoneNumberListInstancePageOptions, callback?: (error: Error | null, items: PhoneNumberPage) => any): Promise<PhoneNumberPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PhoneNumberListInstance(version: V1, serviceSid: string): PhoneNumberListInstance;
export declare class PhoneNumberPage extends Page<V1, PhoneNumberPayload, PhoneNumberResource, PhoneNumberInstance> {
    /**
     * Initialize the PhoneNumberPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: PhoneNumberSolution);
    /**
     * Build an instance of PhoneNumberInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: PhoneNumberResource): PhoneNumberInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
