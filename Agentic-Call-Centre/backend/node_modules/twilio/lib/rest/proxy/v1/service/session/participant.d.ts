/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
import { MessageInteractionListInstance } from "./participant/messageInteraction";
/**
 * Options to pass to create a ParticipantInstance
 */
export interface ParticipantListInstanceCreateOptions {
    /** The phone number of the Participant. */
    identifier: string;
    /** The string that you assigned to describe the participant. This value must be 255 characters or fewer. **This value should not have PII.** */
    friendlyName?: string;
    /** The proxy phone number to use for the Participant. If not specified, Proxy will select a number from the pool. */
    proxyIdentifier?: string;
    /** The SID of the Proxy Identifier to assign to the Participant. */
    proxyIdentifierSid?: string;
}
/**
 * Options to pass to each
 */
export interface ParticipantListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ParticipantInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ParticipantListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ParticipantListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ParticipantContext {
    messageInteractions: MessageInteractionListInstance;
    /**
     * Remove a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    fetch(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ParticipantContextSolution {
    serviceSid: string;
    sessionSid: string;
    sid: string;
}
export declare class ParticipantContextImpl implements ParticipantContext {
    protected _version: V1;
    protected _solution: ParticipantContextSolution;
    protected _uri: string;
    protected _messageInteractions?: MessageInteractionListInstance;
    constructor(_version: V1, serviceSid: string, sessionSid: string, sid: string);
    get messageInteractions(): MessageInteractionListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ParticipantContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ParticipantPayload extends TwilioResponsePayload {
    participants: ParticipantResource[];
}
interface ParticipantResource {
    sid: string;
    session_sid: string;
    service_sid: string;
    account_sid: string;
    friendly_name: string;
    identifier: string;
    proxy_identifier: string;
    proxy_identifier_sid: string;
    date_deleted: Date;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class ParticipantInstance {
    protected _version: V1;
    protected _solution: ParticipantContextSolution;
    protected _context?: ParticipantContext;
    constructor(_version: V1, payload: ParticipantResource, serviceSid: string, sessionSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Participant resource.
     */
    sid: string;
    /**
     * The SID of the parent [Session](https://www.twilio.com/docs/proxy/api/session) resource.
     */
    sessionSid: string;
    /**
     * The SID of the resource\'s parent [Service](https://www.twilio.com/docs/proxy/api/service) resource.
     */
    serviceSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Participant resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the participant. This value must be 255 characters or fewer. Supports UTF-8 characters. **This value should not have PII.**
     */
    friendlyName: string;
    /**
     * The phone number or channel identifier of the Participant. This value must be 191 characters or fewer. Supports UTF-8 characters.
     */
    identifier: string;
    /**
     * The phone number or short code (masked number) of the participant\'s partner. The participant will call or message the partner participant at this number.
     */
    proxyIdentifier: string;
    /**
     * The SID of the Proxy Identifier assigned to the Participant.
     */
    proxyIdentifierSid: string;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date when the Participant was removed from the session.
     */
    dateDeleted: Date;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the resource was created.
     */
    dateCreated: Date;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the resource was last updated.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Participant resource.
     */
    url: string;
    /**
     * The URLs to resources related the participant.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    fetch(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Access the messageInteractions.
     */
    messageInteractions(): MessageInteractionListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        sessionSid: string;
        serviceSid: string;
        accountSid: string;
        friendlyName: string;
        identifier: string;
        proxyIdentifier: string;
        proxyIdentifierSid: string;
        dateDeleted: Date;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ParticipantSolution {
    serviceSid: string;
    sessionSid: string;
}
export interface ParticipantListInstance {
    _version: V1;
    _solution: ParticipantSolution;
    _uri: string;
    (sid: string): ParticipantContext;
    get(sid: string): ParticipantContext;
    /**
     * Create a ParticipantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    create(params: ParticipantListInstanceCreateOptions, callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Streams ParticipantInstance records from the API.
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
     * @param { ParticipantListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ParticipantInstance, done: (err?: Error) => void) => void): void;
    each(params: ParticipantListInstanceEachOptions, callback?: (item: ParticipantInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ParticipantInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ParticipantPage) => any): Promise<ParticipantPage>;
    /**
     * Lists ParticipantInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ParticipantListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ParticipantInstance[]) => any): Promise<ParticipantInstance[]>;
    list(params: ParticipantListInstanceOptions, callback?: (error: Error | null, items: ParticipantInstance[]) => any): Promise<ParticipantInstance[]>;
    /**
     * Retrieve a single page of ParticipantInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ParticipantListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ParticipantPage) => any): Promise<ParticipantPage>;
    page(params: ParticipantListInstancePageOptions, callback?: (error: Error | null, items: ParticipantPage) => any): Promise<ParticipantPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ParticipantListInstance(version: V1, serviceSid: string, sessionSid: string): ParticipantListInstance;
export declare class ParticipantPage extends Page<V1, ParticipantPayload, ParticipantResource, ParticipantInstance> {
    /**
     * Initialize the ParticipantPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ParticipantSolution);
    /**
     * Build an instance of ParticipantInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ParticipantResource): ParticipantInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
