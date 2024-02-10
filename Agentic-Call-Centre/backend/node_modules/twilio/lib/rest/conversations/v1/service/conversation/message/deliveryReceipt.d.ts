/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../../base/Page";
import Response from "../../../../../../http/response";
import V1 from "../../../../V1";
export type DeliveryReceiptDeliveryStatus = "read" | "failed" | "delivered" | "undelivered" | "sent";
/**
 * Options to pass to each
 */
export interface DeliveryReceiptListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: DeliveryReceiptInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface DeliveryReceiptListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface DeliveryReceiptListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface DeliveryReceiptContext {
    /**
     * Fetch a DeliveryReceiptInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeliveryReceiptInstance
     */
    fetch(callback?: (error: Error | null, item?: DeliveryReceiptInstance) => any): Promise<DeliveryReceiptInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface DeliveryReceiptContextSolution {
    chatServiceSid: string;
    conversationSid: string;
    messageSid: string;
    sid: string;
}
export declare class DeliveryReceiptContextImpl implements DeliveryReceiptContext {
    protected _version: V1;
    protected _solution: DeliveryReceiptContextSolution;
    protected _uri: string;
    constructor(_version: V1, chatServiceSid: string, conversationSid: string, messageSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: DeliveryReceiptInstance) => any): Promise<DeliveryReceiptInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): DeliveryReceiptContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface DeliveryReceiptPayload extends TwilioResponsePayload {
    delivery_receipts: DeliveryReceiptResource[];
}
interface DeliveryReceiptResource {
    account_sid: string;
    chat_service_sid: string;
    conversation_sid: string;
    message_sid: string;
    sid: string;
    channel_message_sid: string;
    participant_sid: string;
    status: DeliveryReceiptDeliveryStatus;
    error_code: number;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class DeliveryReceiptInstance {
    protected _version: V1;
    protected _solution: DeliveryReceiptContextSolution;
    protected _context?: DeliveryReceiptContext;
    constructor(_version: V1, payload: DeliveryReceiptResource, chatServiceSid: string, conversationSid: string, messageSid: string, sid?: string);
    /**
     * The unique ID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this participant.
     */
    accountSid: string;
    /**
     * The SID of the [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource) the Message resource is associated with.
     */
    chatServiceSid: string;
    /**
     * The unique ID of the [Conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) for this message.
     */
    conversationSid: string;
    /**
     * The SID of the message within a [Conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) the delivery receipt belongs to
     */
    messageSid: string;
    /**
     * A 34 character string that uniquely identifies this resource.
     */
    sid: string;
    /**
     * A messaging channel-specific identifier for the message delivered to participant e.g. `SMxx` for SMS, `WAxx` for Whatsapp etc.
     */
    channelMessageSid: string;
    /**
     * The unique ID of the participant the delivery receipt belongs to.
     */
    participantSid: string;
    status: DeliveryReceiptDeliveryStatus;
    /**
     * The message [delivery error code](https://www.twilio.com/docs/sms/api/message-resource#delivery-related-errors) for a `failed` status,
     */
    errorCode: number;
    /**
     * The date that this resource was created.
     */
    dateCreated: Date;
    /**
     * The date that this resource was last updated. `null` if the delivery receipt has not been updated.
     */
    dateUpdated: Date;
    /**
     * An absolute API resource URL for this delivery receipt.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a DeliveryReceiptInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeliveryReceiptInstance
     */
    fetch(callback?: (error: Error | null, item?: DeliveryReceiptInstance) => any): Promise<DeliveryReceiptInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        chatServiceSid: string;
        conversationSid: string;
        messageSid: string;
        sid: string;
        channelMessageSid: string;
        participantSid: string;
        status: DeliveryReceiptDeliveryStatus;
        errorCode: number;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DeliveryReceiptSolution {
    chatServiceSid: string;
    conversationSid: string;
    messageSid: string;
}
export interface DeliveryReceiptListInstance {
    _version: V1;
    _solution: DeliveryReceiptSolution;
    _uri: string;
    (sid: string): DeliveryReceiptContext;
    get(sid: string): DeliveryReceiptContext;
    /**
     * Streams DeliveryReceiptInstance records from the API.
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
     * @param { DeliveryReceiptListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: DeliveryReceiptInstance, done: (err?: Error) => void) => void): void;
    each(params: DeliveryReceiptListInstanceEachOptions, callback?: (item: DeliveryReceiptInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of DeliveryReceiptInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: DeliveryReceiptPage) => any): Promise<DeliveryReceiptPage>;
    /**
     * Lists DeliveryReceiptInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DeliveryReceiptListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: DeliveryReceiptInstance[]) => any): Promise<DeliveryReceiptInstance[]>;
    list(params: DeliveryReceiptListInstanceOptions, callback?: (error: Error | null, items: DeliveryReceiptInstance[]) => any): Promise<DeliveryReceiptInstance[]>;
    /**
     * Retrieve a single page of DeliveryReceiptInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DeliveryReceiptListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: DeliveryReceiptPage) => any): Promise<DeliveryReceiptPage>;
    page(params: DeliveryReceiptListInstancePageOptions, callback?: (error: Error | null, items: DeliveryReceiptPage) => any): Promise<DeliveryReceiptPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DeliveryReceiptListInstance(version: V1, chatServiceSid: string, conversationSid: string, messageSid: string): DeliveryReceiptListInstance;
export declare class DeliveryReceiptPage extends Page<V1, DeliveryReceiptPayload, DeliveryReceiptResource, DeliveryReceiptInstance> {
    /**
     * Initialize the DeliveryReceiptPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: DeliveryReceiptSolution);
    /**
     * Build an instance of DeliveryReceiptInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DeliveryReceiptResource): DeliveryReceiptInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
