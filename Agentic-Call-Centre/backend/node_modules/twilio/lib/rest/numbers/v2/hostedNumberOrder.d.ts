/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V2 from "../V2";
import { PhoneNumberCapabilities } from "../../../interfaces";
export type HostedNumberOrderStatus = "received" | "verified" | "pending-loa" | "carrier-processing" | "completed" | "failed" | "action-required";
/**
 * Options to pass to create a HostedNumberOrderInstance
 */
export interface HostedNumberOrderListInstanceCreateOptions {
    /** The number to host in [+E.164](https://en.wikipedia.org/wiki/E.164) format */
    phoneNumber: string;
    /** The contact phone number of the person authorized to sign the Authorization Document. */
    contactPhoneNumber: string;
    /** Optional. A 34 character string that uniquely identifies the Address resource that represents the address of the owner of this phone number. */
    addressSid: string;
    /** Optional. Email of the owner of this phone number that is being hosted. */
    email: string;
    /** This defaults to the AccountSid of the authorization the user is using. This can be provided to specify a subaccount to add the HostedNumberOrder to. */
    accountSid?: string;
    /** A 128 character string that is a human readable text that describes this resource. */
    friendlyName?: string;
    /** Optional. A list of emails that the LOA document for this HostedNumberOrder will be carbon copied to. */
    ccEmails?: Array<string>;
    /** The URL that Twilio should request when somebody sends an SMS to the phone number. This will be copied onto the IncomingPhoneNumber resource. */
    smsUrl?: string;
    /** The HTTP method that should be used to request the SmsUrl. Must be either `GET` or `POST`.  This will be copied onto the IncomingPhoneNumber resource. */
    smsMethod?: string;
    /** A URL that Twilio will request if an error occurs requesting or executing the TwiML defined by SmsUrl. This will be copied onto the IncomingPhoneNumber resource. */
    smsFallbackUrl?: string;
    /** Used to specify that the SMS capability will be hosted on Twilio\\\'s platform. */
    smsCapability?: boolean;
    /** The HTTP method that should be used to request the SmsFallbackUrl. Must be either `GET` or `POST`. This will be copied onto the IncomingPhoneNumber resource. */
    smsFallbackMethod?: string;
    /** Optional. The Status Callback URL attached to the IncomingPhoneNumber resource. */
    statusCallbackUrl?: string;
    /** Optional. The Status Callback Method attached to the IncomingPhoneNumber resource. */
    statusCallbackMethod?: string;
    /** Optional. The 34 character sid of the application Twilio should use to handle SMS messages sent to this number. If a `SmsApplicationSid` is present, Twilio will ignore all of the SMS urls above and use those set on the application. */
    smsApplicationSid?: string;
    /** The title of the person authorized to sign the Authorization Document for this phone number. */
    contactTitle?: string;
}
/**
 * Options to pass to each
 */
export interface HostedNumberOrderListInstanceEachOptions {
    /** The Status of this HostedNumberOrder. One of `received`, `pending-verification`, `verified`, `pending-loa`, `carrier-processing`, `testing`, `completed`, `failed`, or `action-required`. */
    status?: HostedNumberOrderStatus;
    /** Whether the SMS capability will be hosted on our platform. Can be `true` of `false`. */
    smsCapability?: boolean;
    /** An E164 formatted phone number hosted by this HostedNumberOrder. */
    phoneNumber?: string;
    /** A 34 character string that uniquely identifies the IncomingPhoneNumber resource created by this HostedNumberOrder. */
    incomingPhoneNumberSid?: string;
    /** A human readable description of this resource, up to 128 characters. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: HostedNumberOrderInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface HostedNumberOrderListInstanceOptions {
    /** The Status of this HostedNumberOrder. One of `received`, `pending-verification`, `verified`, `pending-loa`, `carrier-processing`, `testing`, `completed`, `failed`, or `action-required`. */
    status?: HostedNumberOrderStatus;
    /** Whether the SMS capability will be hosted on our platform. Can be `true` of `false`. */
    smsCapability?: boolean;
    /** An E164 formatted phone number hosted by this HostedNumberOrder. */
    phoneNumber?: string;
    /** A 34 character string that uniquely identifies the IncomingPhoneNumber resource created by this HostedNumberOrder. */
    incomingPhoneNumberSid?: string;
    /** A human readable description of this resource, up to 128 characters. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface HostedNumberOrderListInstancePageOptions {
    /** The Status of this HostedNumberOrder. One of `received`, `pending-verification`, `verified`, `pending-loa`, `carrier-processing`, `testing`, `completed`, `failed`, or `action-required`. */
    status?: HostedNumberOrderStatus;
    /** Whether the SMS capability will be hosted on our platform. Can be `true` of `false`. */
    smsCapability?: boolean;
    /** An E164 formatted phone number hosted by this HostedNumberOrder. */
    phoneNumber?: string;
    /** A 34 character string that uniquely identifies the IncomingPhoneNumber resource created by this HostedNumberOrder. */
    incomingPhoneNumberSid?: string;
    /** A human readable description of this resource, up to 128 characters. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface HostedNumberOrderContext {
    /**
     * Remove a HostedNumberOrderInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a HostedNumberOrderInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed HostedNumberOrderInstance
     */
    fetch(callback?: (error: Error | null, item?: HostedNumberOrderInstance) => any): Promise<HostedNumberOrderInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface HostedNumberOrderContextSolution {
    sid: string;
}
export declare class HostedNumberOrderContextImpl implements HostedNumberOrderContext {
    protected _version: V2;
    protected _solution: HostedNumberOrderContextSolution;
    protected _uri: string;
    constructor(_version: V2, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: HostedNumberOrderInstance) => any): Promise<HostedNumberOrderInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): HostedNumberOrderContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface HostedNumberOrderPayload extends TwilioResponsePayload {
    items: HostedNumberOrderResource[];
}
interface HostedNumberOrderResource {
    sid: string;
    account_sid: string;
    incoming_phone_number_sid: string;
    address_sid: string;
    signing_document_sid: string;
    phone_number: string;
    capabilities: PhoneNumberCapabilities;
    friendly_name: string;
    status: HostedNumberOrderStatus;
    failure_reason: string;
    date_created: Date;
    date_updated: Date;
    email: string;
    cc_emails: Array<string>;
    url: string;
    contact_title: string;
    contact_phone_number: string;
    bulk_hosting_request_sid: string;
    next_step: string;
}
export declare class HostedNumberOrderInstance {
    protected _version: V2;
    protected _solution: HostedNumberOrderContextSolution;
    protected _context?: HostedNumberOrderContext;
    constructor(_version: V2, payload: HostedNumberOrderResource, sid?: string);
    /**
     * A 34 character string that uniquely identifies this HostedNumberOrder.
     */
    sid: string;
    /**
     * A 34 character string that uniquely identifies the account.
     */
    accountSid: string;
    /**
     * A 34 character string that uniquely identifies the [IncomingPhoneNumber](https://www.twilio.com/docs/phone-numbers/api/incomingphonenumber-resource) resource that represents the phone number being hosted.
     */
    incomingPhoneNumberSid: string;
    /**
     * A 34 character string that uniquely identifies the Address resource that represents the address of the owner of this phone number.
     */
    addressSid: string;
    /**
     * A 34 character string that uniquely identifies the [Authorization Document](https://www.twilio.com/docs/phone-numbers/hosted-numbers/hosted-numbers-api/authorization-document-resource) the user needs to sign.
     */
    signingDocumentSid: string;
    /**
     * Phone number to be hosted. This must be in [E.164](https://en.wikipedia.org/wiki/E.164) format, e.g., +16175551212
     */
    phoneNumber: string;
    capabilities: PhoneNumberCapabilities;
    /**
     * A 128 character string that is a human-readable text that describes this resource.
     */
    friendlyName: string;
    status: HostedNumberOrderStatus;
    /**
     * A message that explains why a hosted_number_order went to status \"action-required\"
     */
    failureReason: string;
    /**
     * The date this resource was created, given as [GMT RFC 2822](http://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date that this resource was updated, given as [GMT RFC 2822](http://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * Email of the owner of this phone number that is being hosted.
     */
    email: string;
    /**
     * A list of emails that LOA document for this HostedNumberOrder will be carbon copied to.
     */
    ccEmails: Array<string>;
    /**
     * The URL of this HostedNumberOrder.
     */
    url: string;
    /**
     * The title of the person authorized to sign the Authorization Document for this phone number.
     */
    contactTitle: string;
    /**
     * The contact phone number of the person authorized to sign the Authorization Document.
     */
    contactPhoneNumber: string;
    /**
     * A 34 character string that uniquely identifies the bulk hosting request associated with this HostedNumberOrder.
     */
    bulkHostingRequestSid: string;
    /**
     * The next step you need to take to complete the hosted number order and request it successfully.
     */
    nextStep: string;
    private get _proxy();
    /**
     * Remove a HostedNumberOrderInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a HostedNumberOrderInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed HostedNumberOrderInstance
     */
    fetch(callback?: (error: Error | null, item?: HostedNumberOrderInstance) => any): Promise<HostedNumberOrderInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        incomingPhoneNumberSid: string;
        addressSid: string;
        signingDocumentSid: string;
        phoneNumber: string;
        capabilities: PhoneNumberCapabilities;
        friendlyName: string;
        status: HostedNumberOrderStatus;
        failureReason: string;
        dateCreated: Date;
        dateUpdated: Date;
        email: string;
        ccEmails: string[];
        url: string;
        contactTitle: string;
        contactPhoneNumber: string;
        bulkHostingRequestSid: string;
        nextStep: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface HostedNumberOrderSolution {
}
export interface HostedNumberOrderListInstance {
    _version: V2;
    _solution: HostedNumberOrderSolution;
    _uri: string;
    (sid: string): HostedNumberOrderContext;
    get(sid: string): HostedNumberOrderContext;
    /**
     * Create a HostedNumberOrderInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed HostedNumberOrderInstance
     */
    create(params: HostedNumberOrderListInstanceCreateOptions, callback?: (error: Error | null, item?: HostedNumberOrderInstance) => any): Promise<HostedNumberOrderInstance>;
    /**
     * Streams HostedNumberOrderInstance records from the API.
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
     * @param { HostedNumberOrderListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: HostedNumberOrderInstance, done: (err?: Error) => void) => void): void;
    each(params: HostedNumberOrderListInstanceEachOptions, callback?: (item: HostedNumberOrderInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of HostedNumberOrderInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: HostedNumberOrderPage) => any): Promise<HostedNumberOrderPage>;
    /**
     * Lists HostedNumberOrderInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { HostedNumberOrderListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: HostedNumberOrderInstance[]) => any): Promise<HostedNumberOrderInstance[]>;
    list(params: HostedNumberOrderListInstanceOptions, callback?: (error: Error | null, items: HostedNumberOrderInstance[]) => any): Promise<HostedNumberOrderInstance[]>;
    /**
     * Retrieve a single page of HostedNumberOrderInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { HostedNumberOrderListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: HostedNumberOrderPage) => any): Promise<HostedNumberOrderPage>;
    page(params: HostedNumberOrderListInstancePageOptions, callback?: (error: Error | null, items: HostedNumberOrderPage) => any): Promise<HostedNumberOrderPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function HostedNumberOrderListInstance(version: V2): HostedNumberOrderListInstance;
export declare class HostedNumberOrderPage extends Page<V2, HostedNumberOrderPayload, HostedNumberOrderResource, HostedNumberOrderInstance> {
    /**
     * Initialize the HostedNumberOrderPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: HostedNumberOrderSolution);
    /**
     * Build an instance of HostedNumberOrderInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: HostedNumberOrderResource): HostedNumberOrderInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
