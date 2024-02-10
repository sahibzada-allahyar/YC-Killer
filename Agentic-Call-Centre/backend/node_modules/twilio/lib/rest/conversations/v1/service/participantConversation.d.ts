/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type ParticipantConversationState = "inactive" | "active" | "closed";
/**
 * Options to pass to each
 */
export interface ParticipantConversationListInstanceEachOptions {
    /** A unique string identifier for the conversation participant as [Conversation User](https://www.twilio.com/docs/conversations/api/user-resource). This parameter is non-null if (and only if) the participant is using the Conversations SDK to communicate. Limited to 256 characters. */
    identity?: string;
    /** A unique string identifier for the conversation participant who\'s not a Conversation User. This parameter could be found in messaging_binding.address field of Participant resource. It should be url-encoded. */
    address?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ParticipantConversationInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ParticipantConversationListInstanceOptions {
    /** A unique string identifier for the conversation participant as [Conversation User](https://www.twilio.com/docs/conversations/api/user-resource). This parameter is non-null if (and only if) the participant is using the Conversations SDK to communicate. Limited to 256 characters. */
    identity?: string;
    /** A unique string identifier for the conversation participant who\'s not a Conversation User. This parameter could be found in messaging_binding.address field of Participant resource. It should be url-encoded. */
    address?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ParticipantConversationListInstancePageOptions {
    /** A unique string identifier for the conversation participant as [Conversation User](https://www.twilio.com/docs/conversations/api/user-resource). This parameter is non-null if (and only if) the participant is using the Conversations SDK to communicate. Limited to 256 characters. */
    identity?: string;
    /** A unique string identifier for the conversation participant who\'s not a Conversation User. This parameter could be found in messaging_binding.address field of Participant resource. It should be url-encoded. */
    address?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ParticipantConversationSolution {
    chatServiceSid: string;
}
export interface ParticipantConversationListInstance {
    _version: V1;
    _solution: ParticipantConversationSolution;
    _uri: string;
    /**
     * Streams ParticipantConversationInstance records from the API.
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
     * @param { ParticipantConversationListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ParticipantConversationInstance, done: (err?: Error) => void) => void): void;
    each(params: ParticipantConversationListInstanceEachOptions, callback?: (item: ParticipantConversationInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ParticipantConversationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ParticipantConversationPage) => any): Promise<ParticipantConversationPage>;
    /**
     * Lists ParticipantConversationInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ParticipantConversationListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ParticipantConversationInstance[]) => any): Promise<ParticipantConversationInstance[]>;
    list(params: ParticipantConversationListInstanceOptions, callback?: (error: Error | null, items: ParticipantConversationInstance[]) => any): Promise<ParticipantConversationInstance[]>;
    /**
     * Retrieve a single page of ParticipantConversationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ParticipantConversationListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ParticipantConversationPage) => any): Promise<ParticipantConversationPage>;
    page(params: ParticipantConversationListInstancePageOptions, callback?: (error: Error | null, items: ParticipantConversationPage) => any): Promise<ParticipantConversationPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ParticipantConversationListInstance(version: V1, chatServiceSid: string): ParticipantConversationListInstance;
interface ParticipantConversationPayload extends TwilioResponsePayload {
    conversations: ParticipantConversationResource[];
}
interface ParticipantConversationResource {
    account_sid: string;
    chat_service_sid: string;
    participant_sid: string;
    participant_user_sid: string;
    participant_identity: string;
    participant_messaging_binding: any;
    conversation_sid: string;
    conversation_unique_name: string;
    conversation_friendly_name: string;
    conversation_attributes: string;
    conversation_date_created: Date;
    conversation_date_updated: Date;
    conversation_created_by: string;
    conversation_state: ParticipantConversationState;
    conversation_timers: any;
    links: Record<string, string>;
}
export declare class ParticipantConversationInstance {
    protected _version: V1;
    constructor(_version: V1, payload: ParticipantConversationResource, chatServiceSid: string);
    /**
     * The unique ID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for this conversation.
     */
    accountSid: string;
    /**
     * The unique ID of the [Conversation Service](https://www.twilio.com/docs/conversations/api/service-resource) this conversation belongs to.
     */
    chatServiceSid: string;
    /**
     * The unique ID of the [Participant](https://www.twilio.com/docs/conversations/api/conversation-participant-resource).
     */
    participantSid: string;
    /**
     * The unique string that identifies the conversation participant as [Conversation User](https://www.twilio.com/docs/conversations/api/user-resource).
     */
    participantUserSid: string;
    /**
     * A unique string identifier for the conversation participant as [Conversation User](https://www.twilio.com/docs/conversations/api/user-resource). This parameter is non-null if (and only if) the participant is using the Conversations SDK to communicate. Limited to 256 characters.
     */
    participantIdentity: string;
    /**
     * Information about how this participant exchanges messages with the conversation. A JSON parameter consisting of type and address fields of the participant.
     */
    participantMessagingBinding: any;
    /**
     * The unique ID of the [Conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) this Participant belongs to.
     */
    conversationSid: string;
    /**
     * An application-defined string that uniquely identifies the Conversation resource.
     */
    conversationUniqueName: string;
    /**
     * The human-readable name of this conversation, limited to 256 characters. Optional.
     */
    conversationFriendlyName: string;
    /**
     * An optional string metadata field you can use to store any data you wish. The string value must contain structurally valid JSON if specified.  **Note** that if the attributes are not set \"{}\" will be returned.
     */
    conversationAttributes: string;
    /**
     * The date that this conversation was created, given in ISO 8601 format.
     */
    conversationDateCreated: Date;
    /**
     * The date that this conversation was last updated, given in ISO 8601 format.
     */
    conversationDateUpdated: Date;
    /**
     * Identity of the creator of this Conversation.
     */
    conversationCreatedBy: string;
    conversationState: ParticipantConversationState;
    /**
     * Timer date values representing state update for this conversation.
     */
    conversationTimers: any;
    /**
     * Contains absolute URLs to access the [participant](https://www.twilio.com/docs/conversations/api/conversation-participant-resource) and [conversation](https://www.twilio.com/docs/conversations/api/conversation-resource) of this conversation.
     */
    links: Record<string, string>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        chatServiceSid: string;
        participantSid: string;
        participantUserSid: string;
        participantIdentity: string;
        participantMessagingBinding: any;
        conversationSid: string;
        conversationUniqueName: string;
        conversationFriendlyName: string;
        conversationAttributes: string;
        conversationDateCreated: Date;
        conversationDateUpdated: Date;
        conversationCreatedBy: string;
        conversationState: ParticipantConversationState;
        conversationTimers: any;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class ParticipantConversationPage extends Page<V1, ParticipantConversationPayload, ParticipantConversationResource, ParticipantConversationInstance> {
    /**
     * Initialize the ParticipantConversationPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ParticipantConversationSolution);
    /**
     * Build an instance of ParticipantConversationInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ParticipantConversationResource): ParticipantConversationInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
