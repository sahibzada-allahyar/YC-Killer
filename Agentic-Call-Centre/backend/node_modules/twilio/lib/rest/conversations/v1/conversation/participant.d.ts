/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type ParticipantWebhookEnabledType = "true" | "false";
/**
 * Options to pass to remove a ParticipantInstance
 */
export interface ParticipantContextRemoveOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: ParticipantWebhookEnabledType;
}
/**
 * Options to pass to update a ParticipantInstance
 */
export interface ParticipantContextUpdateOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: ParticipantWebhookEnabledType;
    /** The date that this resource was created. */
    dateCreated?: Date;
    /** The date that this resource was last updated. */
    dateUpdated?: Date;
    /** An optional string metadata field you can use to store any data you wish. The string value must contain structurally valid JSON if specified.  **Note** that if the attributes are not set \\\"{}\\\" will be returned. */
    attributes?: string;
    /** The SID of a conversation-level [Role](https://www.twilio.com/docs/conversations/api/role-resource) to assign to the participant. */
    roleSid?: string;
    /** The address of the Twilio phone number that the participant is in contact with. \\\'null\\\' value will remove it. */
    "messagingBinding.proxyAddress"?: string;
    /** The address of the Twilio phone number that is used in Group MMS. \\\'null\\\' value will remove it. */
    "messagingBinding.projectedAddress"?: string;
    /** A unique string identifier for the conversation participant as [Conversation User](https://www.twilio.com/docs/conversations/api/user-resource). This parameter is non-null if (and only if) the participant is using the Conversations SDK to communicate. Limited to 256 characters. */
    identity?: string;
    /** Index of last “read” message in the [Conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) for the Participant. */
    lastReadMessageIndex?: number;
    /** Timestamp of last “read” message in the [Conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) for the Participant. */
    lastReadTimestamp?: string;
}
/**
 * Options to pass to create a ParticipantInstance
 */
export interface ParticipantListInstanceCreateOptions {
    /** The X-Twilio-Webhook-Enabled HTTP request header */
    xTwilioWebhookEnabled?: ParticipantWebhookEnabledType;
    /** A unique string identifier for the conversation participant as [Conversation User](https://www.twilio.com/docs/conversations/api/user-resource). This parameter is non-null if (and only if) the participant is using the Conversations SDK to communicate. Limited to 256 characters. */
    identity?: string;
    /** The address of the participant\\\'s device, e.g. a phone or WhatsApp number. Together with the Proxy address, this determines a participant uniquely. This field (with proxy_address) is only null when the participant is interacting from an SDK endpoint (see the \\\'identity\\\' field). */
    "messagingBinding.address"?: string;
    /** The address of the Twilio phone number (or WhatsApp number) that the participant is in contact with. This field, together with participant address, is only null when the participant is interacting from an SDK endpoint (see the \\\'identity\\\' field). */
    "messagingBinding.proxyAddress"?: string;
    /** The date that this resource was created. */
    dateCreated?: Date;
    /** The date that this resource was last updated. */
    dateUpdated?: Date;
    /** An optional string metadata field you can use to store any data you wish. The string value must contain structurally valid JSON if specified.  **Note** that if the attributes are not set \\\"{}\\\" will be returned. */
    attributes?: string;
    /** The address of the Twilio phone number that is used in Group MMS. Communication mask for the Conversation participant with Identity. */
    "messagingBinding.projectedAddress"?: string;
    /** The SID of a conversation-level [Role](https://www.twilio.com/docs/conversations/api/role-resource) to assign to the participant. */
    roleSid?: string;
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
    /**
     * Remove a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a ParticipantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    remove(params: ParticipantContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    fetch(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Update a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    update(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Update a ParticipantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    update(params: ParticipantContextUpdateOptions, callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ParticipantContextSolution {
    conversationSid: string;
    sid: string;
}
export declare class ParticipantContextImpl implements ParticipantContext {
    protected _version: V1;
    protected _solution: ParticipantContextSolution;
    protected _uri: string;
    constructor(_version: V1, conversationSid: string, sid: string);
    remove(params?: ParticipantContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    update(params?: ParticipantContextUpdateOptions | ((error: Error | null, item?: ParticipantInstance) => any), callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
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
    account_sid: string;
    conversation_sid: string;
    sid: string;
    identity: string;
    attributes: string;
    messaging_binding: any;
    role_sid: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    last_read_message_index: number;
    last_read_timestamp: string;
}
export declare class ParticipantInstance {
    protected _version: V1;
    protected _solution: ParticipantContextSolution;
    protected _context?: ParticipantContext;
    constructor(_version: V1, payload: ParticipantResource, conversationSid: string, sid?: string);
    /**
     * The unique ID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this participant.
     */
    accountSid: string;
    /**
     * The unique ID of the [Conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) for this participant.
     */
    conversationSid: string;
    /**
     * A 34 character string that uniquely identifies this resource.
     */
    sid: string;
    /**
     * A unique string identifier for the conversation participant as [Conversation User](https://www.twilio.com/docs/conversations/api/user-resource). This parameter is non-null if (and only if) the participant is using the Conversations SDK to communicate. Limited to 256 characters.
     */
    identity: string;
    /**
     * An optional string metadata field you can use to store any data you wish. The string value must contain structurally valid JSON if specified.  **Note** that if the attributes are not set \"{}\" will be returned.
     */
    attributes: string;
    /**
     * Information about how this participant exchanges messages with the conversation. A JSON parameter consisting of type and address fields of the participant.
     */
    messagingBinding: any;
    /**
     * The SID of a conversation-level [Role](https://www.twilio.com/docs/conversations/api/role-resource) to assign to the participant.
     */
    roleSid: string;
    /**
     * The date that this resource was created.
     */
    dateCreated: Date;
    /**
     * The date that this resource was last updated.
     */
    dateUpdated: Date;
    /**
     * An absolute API resource URL for this participant.
     */
    url: string;
    /**
     * Index of last “read” message in the [Conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) for the Participant.
     */
    lastReadMessageIndex: number;
    /**
     * Timestamp of last “read” message in the [Conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) for the Participant.
     */
    lastReadTimestamp: string;
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
     * Remove a ParticipantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    remove(params: ParticipantContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    fetch(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Update a ParticipantInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    update(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Update a ParticipantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    update(params: ParticipantContextUpdateOptions, callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        conversationSid: string;
        sid: string;
        identity: string;
        attributes: string;
        messagingBinding: any;
        roleSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        lastReadMessageIndex: number;
        lastReadTimestamp: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ParticipantSolution {
    conversationSid: string;
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
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ParticipantInstance
     */
    create(callback?: (error: Error | null, item?: ParticipantInstance) => any): Promise<ParticipantInstance>;
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
export declare function ParticipantListInstance(version: V1, conversationSid: string): ParticipantListInstance;
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
