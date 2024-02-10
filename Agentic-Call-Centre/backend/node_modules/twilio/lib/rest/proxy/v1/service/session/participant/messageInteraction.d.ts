/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../../base/Page";
import Response from "../../../../../../http/response";
import V1 from "../../../../V1";
export type MessageInteractionResourceStatus = "accepted" | "answered" | "busy" | "canceled" | "completed" | "deleted" | "delivered" | "delivery-unknown" | "failed" | "in-progress" | "initiated" | "no-answer" | "queued" | "received" | "receiving" | "ringing" | "scheduled" | "sending" | "sent" | "undelivered" | "unknown";
export type MessageInteractionType = "message" | "voice" | "unknown";
/**
 * Options to pass to create a MessageInteractionInstance
 */
export interface MessageInteractionListInstanceCreateOptions {
    /** The message to send to the participant */
    body?: string;
    /** Reserved. Not currently supported. */
    mediaUrl?: Array<string>;
}
/**
 * Options to pass to each
 */
export interface MessageInteractionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: MessageInteractionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface MessageInteractionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface MessageInteractionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface MessageInteractionContext {
    /**
     * Fetch a MessageInteractionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInteractionInstance
     */
    fetch(callback?: (error: Error | null, item?: MessageInteractionInstance) => any): Promise<MessageInteractionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface MessageInteractionContextSolution {
    serviceSid: string;
    sessionSid: string;
    participantSid: string;
    sid: string;
}
export declare class MessageInteractionContextImpl implements MessageInteractionContext {
    protected _version: V1;
    protected _solution: MessageInteractionContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, sessionSid: string, participantSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: MessageInteractionInstance) => any): Promise<MessageInteractionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): MessageInteractionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface MessageInteractionPayload extends TwilioResponsePayload {
    interactions: MessageInteractionResource[];
}
interface MessageInteractionResource {
    sid: string;
    session_sid: string;
    service_sid: string;
    account_sid: string;
    data: string;
    type: MessageInteractionType;
    participant_sid: string;
    inbound_participant_sid: string;
    inbound_resource_sid: string;
    inbound_resource_status: MessageInteractionResourceStatus;
    inbound_resource_type: string;
    inbound_resource_url: string;
    outbound_participant_sid: string;
    outbound_resource_sid: string;
    outbound_resource_status: MessageInteractionResourceStatus;
    outbound_resource_type: string;
    outbound_resource_url: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class MessageInteractionInstance {
    protected _version: V1;
    protected _solution: MessageInteractionContextSolution;
    protected _context?: MessageInteractionContext;
    constructor(_version: V1, payload: MessageInteractionResource, serviceSid: string, sessionSid: string, participantSid: string, sid?: string);
    /**
     * The unique string that we created to identify the MessageInteraction resource.
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
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the MessageInteraction resource.
     */
    accountSid: string;
    /**
     * A JSON string that includes the message body sent to the participant. (e.g. `{\"body\": \"hello\"}`)
     */
    data: string;
    type: MessageInteractionType;
    /**
     * The SID of the [Participant](https://www.twilio.com/docs/proxy/api/participant) resource.
     */
    participantSid: string;
    /**
     * Always empty for created Message Interactions.
     */
    inboundParticipantSid: string;
    /**
     * Always empty for created Message Interactions.
     */
    inboundResourceSid: string;
    inboundResourceStatus: MessageInteractionResourceStatus;
    /**
     * Always empty for created Message Interactions.
     */
    inboundResourceType: string;
    /**
     * Always empty for created Message Interactions.
     */
    inboundResourceUrl: string;
    /**
     * The SID of the outbound [Participant](https://www.twilio.com/docs/proxy/api/participant) resource.
     */
    outboundParticipantSid: string;
    /**
     * The SID of the outbound [Message](https://www.twilio.com/docs/sms/api/message-resource) resource.
     */
    outboundResourceSid: string;
    outboundResourceStatus: MessageInteractionResourceStatus;
    /**
     * The outbound resource type. This value is always `Message`.
     */
    outboundResourceType: string;
    /**
     * The URL of the Twilio message resource.
     */
    outboundResourceUrl: string;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the resource was created.
     */
    dateCreated: Date;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the resource was last updated.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the MessageInteraction resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a MessageInteractionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInteractionInstance
     */
    fetch(callback?: (error: Error | null, item?: MessageInteractionInstance) => any): Promise<MessageInteractionInstance>;
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
        type: MessageInteractionType;
        participantSid: string;
        inboundParticipantSid: string;
        inboundResourceSid: string;
        inboundResourceStatus: MessageInteractionResourceStatus;
        inboundResourceType: string;
        inboundResourceUrl: string;
        outboundParticipantSid: string;
        outboundResourceSid: string;
        outboundResourceStatus: MessageInteractionResourceStatus;
        outboundResourceType: string;
        outboundResourceUrl: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface MessageInteractionSolution {
    serviceSid: string;
    sessionSid: string;
    participantSid: string;
}
export interface MessageInteractionListInstance {
    _version: V1;
    _solution: MessageInteractionSolution;
    _uri: string;
    (sid: string): MessageInteractionContext;
    get(sid: string): MessageInteractionContext;
    /**
     * Create a MessageInteractionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInteractionInstance
     */
    create(callback?: (error: Error | null, item?: MessageInteractionInstance) => any): Promise<MessageInteractionInstance>;
    /**
     * Create a MessageInteractionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed MessageInteractionInstance
     */
    create(params: MessageInteractionListInstanceCreateOptions, callback?: (error: Error | null, item?: MessageInteractionInstance) => any): Promise<MessageInteractionInstance>;
    /**
     * Streams MessageInteractionInstance records from the API.
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
     * @param { MessageInteractionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: MessageInteractionInstance, done: (err?: Error) => void) => void): void;
    each(params: MessageInteractionListInstanceEachOptions, callback?: (item: MessageInteractionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of MessageInteractionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: MessageInteractionPage) => any): Promise<MessageInteractionPage>;
    /**
     * Lists MessageInteractionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MessageInteractionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: MessageInteractionInstance[]) => any): Promise<MessageInteractionInstance[]>;
    list(params: MessageInteractionListInstanceOptions, callback?: (error: Error | null, items: MessageInteractionInstance[]) => any): Promise<MessageInteractionInstance[]>;
    /**
     * Retrieve a single page of MessageInteractionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { MessageInteractionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: MessageInteractionPage) => any): Promise<MessageInteractionPage>;
    page(params: MessageInteractionListInstancePageOptions, callback?: (error: Error | null, items: MessageInteractionPage) => any): Promise<MessageInteractionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function MessageInteractionListInstance(version: V1, serviceSid: string, sessionSid: string, participantSid: string): MessageInteractionListInstance;
export declare class MessageInteractionPage extends Page<V1, MessageInteractionPayload, MessageInteractionResource, MessageInteractionInstance> {
    /**
     * Initialize the MessageInteractionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: MessageInteractionSolution);
    /**
     * Build an instance of MessageInteractionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: MessageInteractionResource): MessageInteractionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
