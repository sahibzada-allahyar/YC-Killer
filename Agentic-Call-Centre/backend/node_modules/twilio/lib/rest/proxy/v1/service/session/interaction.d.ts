/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
export type InteractionResourceStatus = "accepted" | "answered" | "busy" | "canceled" | "completed" | "deleted" | "delivered" | "delivery-unknown" | "failed" | "in-progress" | "initiated" | "no-answer" | "queued" | "received" | "receiving" | "ringing" | "scheduled" | "sending" | "sent" | "undelivered" | "unknown";
export type InteractionType = "message" | "voice" | "unknown";
/**
 * Options to pass to each
 */
export interface InteractionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: InteractionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface InteractionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface InteractionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface InteractionContext {
    /**
     * Remove a InteractionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a InteractionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionInstance
     */
    fetch(callback?: (error: Error | null, item?: InteractionInstance) => any): Promise<InteractionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface InteractionContextSolution {
    serviceSid: string;
    sessionSid: string;
    sid: string;
}
export declare class InteractionContextImpl implements InteractionContext {
    protected _version: V1;
    protected _solution: InteractionContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, sessionSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: InteractionInstance) => any): Promise<InteractionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): InteractionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface InteractionPayload extends TwilioResponsePayload {
    interactions: InteractionResource[];
}
interface InteractionResource {
    sid: string;
    session_sid: string;
    service_sid: string;
    account_sid: string;
    data: string;
    type: InteractionType;
    inbound_participant_sid: string;
    inbound_resource_sid: string;
    inbound_resource_status: InteractionResourceStatus;
    inbound_resource_type: string;
    inbound_resource_url: string;
    outbound_participant_sid: string;
    outbound_resource_sid: string;
    outbound_resource_status: InteractionResourceStatus;
    outbound_resource_type: string;
    outbound_resource_url: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class InteractionInstance {
    protected _version: V1;
    protected _solution: InteractionContextSolution;
    protected _context?: InteractionContext;
    constructor(_version: V1, payload: InteractionResource, serviceSid: string, sessionSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Interaction resource.
     */
    sid: string;
    /**
     * The SID of the parent [Session](https://www.twilio.com/docs/proxy/api/session) resource.
     */
    sessionSid: string;
    /**
     * The SID of the parent [Service](https://www.twilio.com/docs/proxy/api/service) resource.
     */
    serviceSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Interaction resource.
     */
    accountSid: string;
    /**
     * A JSON string that includes the message body of message interactions (e.g. `{\"body\": \"hello\"}`) or the call duration (when available) of a call (e.g. `{\"duration\": \"5\"}`).
     */
    data: string;
    type: InteractionType;
    /**
     * The SID of the inbound [Participant](https://www.twilio.com/docs/proxy/api/participant) resource.
     */
    inboundParticipantSid: string;
    /**
     * The SID of the inbound resource; either the [Call](https://www.twilio.com/docs/voice/api/call-resource) or [Message](https://www.twilio.com/docs/sms/api/message-resource).
     */
    inboundResourceSid: string;
    inboundResourceStatus: InteractionResourceStatus;
    /**
     * The inbound resource type. Can be [Call](https://www.twilio.com/docs/voice/api/call-resource) or [Message](https://www.twilio.com/docs/sms/api/message-resource).
     */
    inboundResourceType: string;
    /**
     * The URL of the Twilio inbound resource
     */
    inboundResourceUrl: string;
    /**
     * The SID of the outbound [Participant](https://www.twilio.com/docs/proxy/api/participant)).
     */
    outboundParticipantSid: string;
    /**
     * The SID of the outbound resource; either the [Call](https://www.twilio.com/docs/voice/api/call-resource) or [Message](https://www.twilio.com/docs/sms/api/message-resource).
     */
    outboundResourceSid: string;
    outboundResourceStatus: InteractionResourceStatus;
    /**
     * The outbound resource type. Can be: [Call](https://www.twilio.com/docs/voice/api/call-resource) or [Message](https://www.twilio.com/docs/sms/api/message-resource).
     */
    outboundResourceType: string;
    /**
     * The URL of the Twilio outbound resource.
     */
    outboundResourceUrl: string;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the Interaction was created.
     */
    dateCreated: Date;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the resource was last updated.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Interaction resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a InteractionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a InteractionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionInstance
     */
    fetch(callback?: (error: Error | null, item?: InteractionInstance) => any): Promise<InteractionInstance>;
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
        data: string;
        type: InteractionType;
        inboundParticipantSid: string;
        inboundResourceSid: string;
        inboundResourceStatus: InteractionResourceStatus;
        inboundResourceType: string;
        inboundResourceUrl: string;
        outboundParticipantSid: string;
        outboundResourceSid: string;
        outboundResourceStatus: InteractionResourceStatus;
        outboundResourceType: string;
        outboundResourceUrl: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface InteractionSolution {
    serviceSid: string;
    sessionSid: string;
}
export interface InteractionListInstance {
    _version: V1;
    _solution: InteractionSolution;
    _uri: string;
    (sid: string): InteractionContext;
    get(sid: string): InteractionContext;
    /**
     * Streams InteractionInstance records from the API.
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
     * @param { InteractionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: InteractionInstance, done: (err?: Error) => void) => void): void;
    each(params: InteractionListInstanceEachOptions, callback?: (item: InteractionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of InteractionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: InteractionPage) => any): Promise<InteractionPage>;
    /**
     * Lists InteractionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InteractionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: InteractionInstance[]) => any): Promise<InteractionInstance[]>;
    list(params: InteractionListInstanceOptions, callback?: (error: Error | null, items: InteractionInstance[]) => any): Promise<InteractionInstance[]>;
    /**
     * Retrieve a single page of InteractionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InteractionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: InteractionPage) => any): Promise<InteractionPage>;
    page(params: InteractionListInstancePageOptions, callback?: (error: Error | null, items: InteractionPage) => any): Promise<InteractionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InteractionListInstance(version: V1, serviceSid: string, sessionSid: string): InteractionListInstance;
export declare class InteractionPage extends Page<V1, InteractionPayload, InteractionResource, InteractionInstance> {
    /**
     * Initialize the InteractionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: InteractionSolution);
    /**
     * Build an instance of InteractionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: InteractionResource): InteractionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
