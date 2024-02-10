/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import HostedNumbers from "../../HostedNumbers";
import { PhoneNumberCapabilities } from "../../../../interfaces";
export type DependentHostedNumberOrderStatus = "received" | "pending-verification" | "verified" | "pending-loa" | "carrier-processing" | "testing" | "completed" | "failed" | "action-required";
export type DependentHostedNumberOrderVerificationType = "phone-call" | "phone-bill";
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
    /** A human readable description of this resource, up to 64 characters. */
    friendlyName?: string;
    /** Provides a unique and addressable name to be assigned to this HostedNumberOrder, assigned by the developer, to be optionally used in addition to SID. */
    uniqueName?: string;
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
    /** A human readable description of this resource, up to 64 characters. */
    friendlyName?: string;
    /** Provides a unique and addressable name to be assigned to this HostedNumberOrder, assigned by the developer, to be optionally used in addition to SID. */
    uniqueName?: string;
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
    /** A human readable description of this resource, up to 64 characters. */
    friendlyName?: string;
    /** Provides a unique and addressable name to be assigned to this HostedNumberOrder, assigned by the developer, to be optionally used in addition to SID. */
    uniqueName?: string;
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
    _version: HostedNumbers;
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
export declare function DependentHostedNumberOrderListInstance(version: HostedNumbers, signingDocumentSid: string): DependentHostedNumberOrderListInstance;
interface DependentHostedNumberOrderPayload extends TwilioResponsePayload {
    items: DependentHostedNumberOrderResource[];
}
interface DependentHostedNumberOrderResource {
    sid: string;
    account_sid: string;
    incoming_phone_number_sid: string;
    address_sid: string;
    signing_document_sid: string;
    phone_number: string;
    capabilities: PhoneNumberCapabilities;
    friendly_name: string;
    unique_name: string;
    status: DependentHostedNumberOrderStatus;
    failure_reason: string;
    date_created: Date;
    date_updated: Date;
    verification_attempts: number;
    email: string;
    cc_emails: Array<string>;
    verification_type: DependentHostedNumberOrderVerificationType;
    verification_document_sid: string;
    extension: string;
    call_delay: number;
    verification_code: string;
    verification_call_sids: Array<string>;
}
export declare class DependentHostedNumberOrderInstance {
    protected _version: HostedNumbers;
    constructor(_version: HostedNumbers, payload: DependentHostedNumberOrderResource, signingDocumentSid: string);
    /**
     * A 34 character string that uniquely identifies this Authorization Document
     */
    sid: string;
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
     * A human readable description of this resource, up to 64 characters.
     */
    friendlyName: string;
    /**
     * Provides a unique and addressable name to be assigned to this HostedNumberOrder, assigned by the developer, to be optionally used in addition to SID.
     */
    uniqueName: string;
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
     * The number of attempts made to verify ownership of the phone number that is being hosted.
     */
    verificationAttempts: number;
    /**
     * Email of the owner of this phone number that is being hosted.
     */
    email: string;
    /**
     * Email recipients who will be informed when an Authorization Document has been sent and signed
     */
    ccEmails: Array<string>;
    verificationType: DependentHostedNumberOrderVerificationType;
    /**
     * A 34 character string that uniquely identifies the Identity Document resource that represents the document for verifying ownership of the number to be hosted.
     */
    verificationDocumentSid: string;
    /**
     * A numerical extension to be used when making the ownership verification call.
     */
    extension: string;
    /**
     * A value between 0-30 specifying the number of seconds to delay initiating the ownership verification call.
     */
    callDelay: number;
    /**
     * The digits passed during the ownership verification call.
     */
    verificationCode: string;
    /**
     * A list of 34 character strings that are unique identifiers for the calls placed as part of ownership verification.
     */
    verificationCallSids: Array<string>;
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
        uniqueName: string;
        status: DependentHostedNumberOrderStatus;
        failureReason: string;
        dateCreated: Date;
        dateUpdated: Date;
        verificationAttempts: number;
        email: string;
        ccEmails: string[];
        verificationType: DependentHostedNumberOrderVerificationType;
        verificationDocumentSid: string;
        extension: string;
        callDelay: number;
        verificationCode: string;
        verificationCallSids: string[];
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class DependentHostedNumberOrderPage extends Page<HostedNumbers, DependentHostedNumberOrderPayload, DependentHostedNumberOrderResource, DependentHostedNumberOrderInstance> {
    /**
     * Initialize the DependentHostedNumberOrderPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: HostedNumbers, response: Response<string>, solution: DependentHostedNumberOrderSolution);
    /**
     * Build an instance of DependentHostedNumberOrderInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DependentHostedNumberOrderResource): DependentHostedNumberOrderInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
