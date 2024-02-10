/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
import { PhoneNumberCapabilities } from "../../../../interfaces";
export type DependentHostedNumberOrderStatus = "received" | "verified" | "pending-loa" | "carrier-processing" | "completed" | "failed" | "action-required";
/**
 * Options to pass to each
 */
export interface DependentHostedNumberOrderListInstanceEachOptions {
    /** Status of an instance resource. It can hold one of the values: 1. opened 2. signing, 3. signed LOA, 4. canceled, 5. failed. See the section entitled [Status Values](https://www.twilio.com/docs/phone-numbers/hosted-numbers/hosted-numbers-api/authorization-document-resource#status-values) for more information on each of these statuses. */
    status?: DependentHostedNumberOrderStatus;
    /** An E164 formatted phone number hosted by this HostedNumberOrder. */
    phoneNumber?: string;
    /** A 34 character string that uniquely identifies the IncomingPhoneNumber resource created by this HostedNumberOrder. */
    incomingPhoneNumberSid?: string;
    /** A human readable description of this resource, up to 128 characters. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: DependentHostedNumberOrderInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface DependentHostedNumberOrderListInstanceOptions {
    /** Status of an instance resource. It can hold one of the values: 1. opened 2. signing, 3. signed LOA, 4. canceled, 5. failed. See the section entitled [Status Values](https://www.twilio.com/docs/phone-numbers/hosted-numbers/hosted-numbers-api/authorization-document-resource#status-values) for more information on each of these statuses. */
    status?: DependentHostedNumberOrderStatus;
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
export interface DependentHostedNumberOrderListInstancePageOptions {
    /** Status of an instance resource. It can hold one of the values: 1. opened 2. signing, 3. signed LOA, 4. canceled, 5. failed. See the section entitled [Status Values](https://www.twilio.com/docs/phone-numbers/hosted-numbers/hosted-numbers-api/authorization-document-resource#status-values) for more information on each of these statuses. */
    status?: DependentHostedNumberOrderStatus;
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
export interface DependentHostedNumberOrderSolution {
    signingDocumentSid: string;
}
export interface DependentHostedNumberOrderListInstance {
    _version: V2;
    _solution: DependentHostedNumberOrderSolution;
    _uri: string;
    /**
     * Streams DependentHostedNumberOrderInstance records from the API.
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
     * @param { DependentHostedNumberOrderListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: DependentHostedNumberOrderInstance, done: (err?: Error) => void) => void): void;
    each(params: DependentHostedNumberOrderListInstanceEachOptions, callback?: (item: DependentHostedNumberOrderInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of DependentHostedNumberOrderInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: DependentHostedNumberOrderPage) => any): Promise<DependentHostedNumberOrderPage>;
    /**
     * Lists DependentHostedNumberOrderInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DependentHostedNumberOrderListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: DependentHostedNumberOrderInstance[]) => any): Promise<DependentHostedNumberOrderInstance[]>;
    list(params: DependentHostedNumberOrderListInstanceOptions, callback?: (error: Error | null, items: DependentHostedNumberOrderInstance[]) => any): Promise<DependentHostedNumberOrderInstance[]>;
    /**
     * Retrieve a single page of DependentHostedNumberOrderInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DependentHostedNumberOrderListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: DependentHostedNumberOrderPage) => any): Promise<DependentHostedNumberOrderPage>;
    page(params: DependentHostedNumberOrderListInstancePageOptions, callback?: (error: Error | null, items: DependentHostedNumberOrderPage) => any): Promise<DependentHostedNumberOrderPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DependentHostedNumberOrderListInstance(version: V2, signingDocumentSid: string): DependentHostedNumberOrderListInstance;
interface DependentHostedNumberOrderPayload extends TwilioResponsePayload {
    items: DependentHostedNumberOrderResource[];
}
interface DependentHostedNumberOrderResource {
    sid: string;
    bulk_hosting_request_sid: string;
    next_step: string;
    account_sid: string;
    incoming_phone_number_sid: string;
    address_sid: string;
    signing_document_sid: string;
    phone_number: string;
    capabilities: PhoneNumberCapabilities;
    friendly_name: string;
    status: DependentHostedNumberOrderStatus;
    failure_reason: string;
    date_created: Date;
    date_updated: Date;
    email: string;
    cc_emails: Array<string>;
    contact_title: string;
    contact_phone_number: string;
}
export declare class DependentHostedNumberOrderInstance {
    protected _version: V2;
    constructor(_version: V2, payload: DependentHostedNumberOrderResource, signingDocumentSid: string);
    /**
     * A 34 character string that uniquely identifies this Authorization Document
     */
    sid: string;
    /**
     * A 34 character string that uniquely identifies the bulk hosting request associated with this HostedNumberOrder.
     */
    bulkHostingRequestSid: string;
    /**
     * The next step you need to take to complete the hosted number order and request it successfully.
     */
    nextStep: string;
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * A 34 character string that uniquely identifies the IncomingPhoneNumber resource created by this HostedNumberOrder.
     */
    incomingPhoneNumberSid: string;
    /**
     * A 34 character string that uniquely identifies the Address resource that represents the address of the owner of this phone number.
     */
    addressSid: string;
    /**
     * A 34 character string that uniquely identifies the LOA document associated with this HostedNumberOrder.
     */
    signingDocumentSid: string;
    /**
     * An E164 formatted phone number hosted by this HostedNumberOrder.
     */
    phoneNumber: string;
    capabilities: PhoneNumberCapabilities;
    /**
     * A human readable description of this resource, up to 128 characters.
     */
    friendlyName: string;
    status: DependentHostedNumberOrderStatus;
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
     * Email recipients who will be informed when an Authorization Document has been sent and signed
     */
    ccEmails: Array<string>;
    /**
     * The title of the person authorized to sign the Authorization Document for this phone number.
     */
    contactTitle: string;
    /**
     * The contact phone number of the person authorized to sign the Authorization Document.
     */
    contactPhoneNumber: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        bulkHostingRequestSid: string;
        nextStep: string;
        accountSid: string;
        incomingPhoneNumberSid: string;
        addressSid: string;
        signingDocumentSid: string;
        phoneNumber: string;
        capabilities: PhoneNumberCapabilities;
        friendlyName: string;
        status: DependentHostedNumberOrderStatus;
        failureReason: string;
        dateCreated: Date;
        dateUpdated: Date;
        email: string;
        ccEmails: string[];
        contactTitle: string;
        contactPhoneNumber: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class DependentHostedNumberOrderPage extends Page<V2, DependentHostedNumberOrderPayload, DependentHostedNumberOrderResource, DependentHostedNumberOrderInstance> {
    /**
     * Initialize the DependentHostedNumberOrderPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: DependentHostedNumberOrderSolution);
    /**
     * Build an instance of DependentHostedNumberOrderInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DependentHostedNumberOrderResource): DependentHostedNumberOrderInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
