/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V2 from "../V2";
import { DependentHostedNumberOrderListInstance } from "./authorizationDocument/dependentHostedNumberOrder";
export type AuthorizationDocumentStatus = "opened" | "signing" | "signed" | "canceled" | "failed";
/**
 * Options to pass to create a AuthorizationDocumentInstance
 */
export interface AuthorizationDocumentListInstanceCreateOptions {
    /** A 34 character string that uniquely identifies the Address resource that is associated with this AuthorizationDocument. */
    addressSid: string;
    /** Email that this AuthorizationDocument will be sent to for signing. */
    email: string;
    /** The contact phone number of the person authorized to sign the Authorization Document. */
    contactPhoneNumber: string;
    /** A list of HostedNumberOrder sids that this AuthorizationDocument will authorize for hosting phone number capabilities on Twilio\\\'s platform. */
    hostedNumberOrderSids: Array<string>;
    /** The title of the person authorized to sign the Authorization Document for this phone number. */
    contactTitle?: string;
    /** Email recipients who will be informed when an Authorization Document has been sent and signed. */
    ccEmails?: Array<string>;
}
/**
 * Options to pass to each
 */
export interface AuthorizationDocumentListInstanceEachOptions {
    /** Email that this AuthorizationDocument will be sent to for signing. */
    email?: string;
    /** Status of an instance resource. It can hold one of the values: 1. opened 2. signing, 3. signed LOA, 4. canceled, 5. failed. See the section entitled [Status Values](https://www.twilio.com/docs/phone-numbers/hosted-numbers/hosted-numbers-api/authorization-document-resource#status-values) for more information on each of these statuses. */
    status?: AuthorizationDocumentStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AuthorizationDocumentInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AuthorizationDocumentListInstanceOptions {
    /** Email that this AuthorizationDocument will be sent to for signing. */
    email?: string;
    /** Status of an instance resource. It can hold one of the values: 1. opened 2. signing, 3. signed LOA, 4. canceled, 5. failed. See the section entitled [Status Values](https://www.twilio.com/docs/phone-numbers/hosted-numbers/hosted-numbers-api/authorization-document-resource#status-values) for more information on each of these statuses. */
    status?: AuthorizationDocumentStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AuthorizationDocumentListInstancePageOptions {
    /** Email that this AuthorizationDocument will be sent to for signing. */
    email?: string;
    /** Status of an instance resource. It can hold one of the values: 1. opened 2. signing, 3. signed LOA, 4. canceled, 5. failed. See the section entitled [Status Values](https://www.twilio.com/docs/phone-numbers/hosted-numbers/hosted-numbers-api/authorization-document-resource#status-values) for more information on each of these statuses. */
    status?: AuthorizationDocumentStatus;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AuthorizationDocumentContext {
    dependentHostedNumberOrders: DependentHostedNumberOrderListInstance;
    /**
     * Remove a AuthorizationDocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AuthorizationDocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthorizationDocumentInstance
     */
    fetch(callback?: (error: Error | null, item?: AuthorizationDocumentInstance) => any): Promise<AuthorizationDocumentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AuthorizationDocumentContextSolution {
    sid: string;
}
export declare class AuthorizationDocumentContextImpl implements AuthorizationDocumentContext {
    protected _version: V2;
    protected _solution: AuthorizationDocumentContextSolution;
    protected _uri: string;
    protected _dependentHostedNumberOrders?: DependentHostedNumberOrderListInstance;
    constructor(_version: V2, sid: string);
    get dependentHostedNumberOrders(): DependentHostedNumberOrderListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: AuthorizationDocumentInstance) => any): Promise<AuthorizationDocumentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AuthorizationDocumentContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AuthorizationDocumentPayload extends TwilioResponsePayload {
    items: AuthorizationDocumentResource[];
}
interface AuthorizationDocumentResource {
    sid: string;
    address_sid: string;
    status: AuthorizationDocumentStatus;
    email: string;
    cc_emails: Array<string>;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class AuthorizationDocumentInstance {
    protected _version: V2;
    protected _solution: AuthorizationDocumentContextSolution;
    protected _context?: AuthorizationDocumentContext;
    constructor(_version: V2, payload: AuthorizationDocumentResource, sid?: string);
    /**
     * A 34 character string that uniquely identifies this AuthorizationDocument.
     */
    sid: string;
    /**
     * A 34 character string that uniquely identifies the Address resource that is associated with this AuthorizationDocument.
     */
    addressSid: string;
    status: AuthorizationDocumentStatus;
    /**
     * Email that this AuthorizationDocument will be sent to for signing.
     */
    email: string;
    /**
     * Email recipients who will be informed when an Authorization Document has been sent and signed.
     */
    ccEmails: Array<string>;
    /**
     * The date this resource was created, given as [GMT RFC 2822](http://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date that this resource was updated, given as [GMT RFC 2822](http://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    url: string;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a AuthorizationDocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a AuthorizationDocumentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthorizationDocumentInstance
     */
    fetch(callback?: (error: Error | null, item?: AuthorizationDocumentInstance) => any): Promise<AuthorizationDocumentInstance>;
    /**
     * Access the dependentHostedNumberOrders.
     */
    dependentHostedNumberOrders(): DependentHostedNumberOrderListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        addressSid: string;
        status: AuthorizationDocumentStatus;
        email: string;
        ccEmails: string[];
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AuthorizationDocumentSolution {
}
export interface AuthorizationDocumentListInstance {
    _version: V2;
    _solution: AuthorizationDocumentSolution;
    _uri: string;
    (sid: string): AuthorizationDocumentContext;
    get(sid: string): AuthorizationDocumentContext;
    /**
     * Create a AuthorizationDocumentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AuthorizationDocumentInstance
     */
    create(params: AuthorizationDocumentListInstanceCreateOptions, callback?: (error: Error | null, item?: AuthorizationDocumentInstance) => any): Promise<AuthorizationDocumentInstance>;
    /**
     * Streams AuthorizationDocumentInstance records from the API.
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
     * @param { AuthorizationDocumentListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AuthorizationDocumentInstance, done: (err?: Error) => void) => void): void;
    each(params: AuthorizationDocumentListInstanceEachOptions, callback?: (item: AuthorizationDocumentInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AuthorizationDocumentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AuthorizationDocumentPage) => any): Promise<AuthorizationDocumentPage>;
    /**
     * Lists AuthorizationDocumentInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AuthorizationDocumentListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AuthorizationDocumentInstance[]) => any): Promise<AuthorizationDocumentInstance[]>;
    list(params: AuthorizationDocumentListInstanceOptions, callback?: (error: Error | null, items: AuthorizationDocumentInstance[]) => any): Promise<AuthorizationDocumentInstance[]>;
    /**
     * Retrieve a single page of AuthorizationDocumentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AuthorizationDocumentListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AuthorizationDocumentPage) => any): Promise<AuthorizationDocumentPage>;
    page(params: AuthorizationDocumentListInstancePageOptions, callback?: (error: Error | null, items: AuthorizationDocumentPage) => any): Promise<AuthorizationDocumentPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AuthorizationDocumentListInstance(version: V2): AuthorizationDocumentListInstance;
export declare class AuthorizationDocumentPage extends Page<V2, AuthorizationDocumentPayload, AuthorizationDocumentResource, AuthorizationDocumentInstance> {
    /**
     * Initialize the AuthorizationDocumentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: AuthorizationDocumentSolution);
    /**
     * Build an instance of AuthorizationDocumentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AuthorizationDocumentResource): AuthorizationDocumentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
