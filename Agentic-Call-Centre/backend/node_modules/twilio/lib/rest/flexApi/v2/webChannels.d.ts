/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../V2";
/**
 * Options to pass to create a WebChannelsInstance
 */
export interface WebChannelsListInstanceCreateOptions {
    /** The SID of the Conversations Address. See [Address Configuration Resource](https://www.twilio.com/docs/conversations/api/address-configuration-resource) for configuration details. When a conversation is created on the Flex backend, the callback URL will be set to the corresponding Studio Flow SID or webhook URL in your address configuration. */
    addressSid: string;
    /** The Conversation\\\'s friendly name. See the [Conversation resource](https://www.twilio.com/docs/conversations/api/conversation-resource) for an example. */
    chatFriendlyName?: string;
    /** The Conversation participant\\\'s friendly name. See the [Conversation Participant Resource](https://www.twilio.com/docs/conversations/api/conversation-participant-resource) for an example. */
    customerFriendlyName?: string;
    /** The pre-engagement data. */
    preEngagementData?: string;
}
export interface WebChannelsSolution {
}
export interface WebChannelsListInstance {
    _version: V2;
    _solution: WebChannelsSolution;
    _uri: string;
    /**
     * Create a WebChannelsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WebChannelsInstance
     */
    create(params: WebChannelsListInstanceCreateOptions, callback?: (error: Error | null, item?: WebChannelsInstance) => any): Promise<WebChannelsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WebChannelsListInstance(version: V2): WebChannelsListInstance;
interface WebChannelsResource {
    conversation_sid: string;
    identity: string;
}
export declare class WebChannelsInstance {
    protected _version: V2;
    constructor(_version: V2, payload: WebChannelsResource);
    /**
     * The unique string representing the [Conversation resource](https://www.twilio.com/docs/conversations/api/conversation-resource) created.
     */
    conversationSid: string;
    /**
     * The unique string representing the User created and should be authorized to participate in the Conversation. For more details, see [User Identity & Access Tokens](https://www.twilio.com/docs/conversations/identity).
     */
    identity: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        conversationSid: string;
        identity: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
