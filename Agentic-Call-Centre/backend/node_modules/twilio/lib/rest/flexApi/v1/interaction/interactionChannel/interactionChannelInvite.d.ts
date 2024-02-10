/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
/**
 * Options to pass to create a InteractionChannelInviteInstance
 */
export interface InteractionChannelInviteListInstanceCreateOptions {
    /** The Interaction\\\'s routing logic. */
    routing: any;
}
/**
 * Options to pass to each
 */
export interface InteractionChannelInviteListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: InteractionChannelInviteInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface InteractionChannelInviteListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface InteractionChannelInviteListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface InteractionChannelInviteSolution {
    interactionSid: string;
    channelSid: string;
}
export interface InteractionChannelInviteListInstance {
    _version: V1;
    _solution: InteractionChannelInviteSolution;
    _uri: string;
    /**
     * Create a InteractionChannelInviteInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionChannelInviteInstance
     */
    create(params: InteractionChannelInviteListInstanceCreateOptions, callback?: (error: Error | null, item?: InteractionChannelInviteInstance) => any): Promise<InteractionChannelInviteInstance>;
    /**
     * Streams InteractionChannelInviteInstance records from the API.
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
     * @param { InteractionChannelInviteListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: InteractionChannelInviteInstance, done: (err?: Error) => void) => void): void;
    each(params: InteractionChannelInviteListInstanceEachOptions, callback?: (item: InteractionChannelInviteInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of InteractionChannelInviteInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: InteractionChannelInvitePage) => any): Promise<InteractionChannelInvitePage>;
    /**
     * Lists InteractionChannelInviteInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InteractionChannelInviteListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: InteractionChannelInviteInstance[]) => any): Promise<InteractionChannelInviteInstance[]>;
    list(params: InteractionChannelInviteListInstanceOptions, callback?: (error: Error | null, items: InteractionChannelInviteInstance[]) => any): Promise<InteractionChannelInviteInstance[]>;
    /**
     * Retrieve a single page of InteractionChannelInviteInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InteractionChannelInviteListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: InteractionChannelInvitePage) => any): Promise<InteractionChannelInvitePage>;
    page(params: InteractionChannelInviteListInstancePageOptions, callback?: (error: Error | null, items: InteractionChannelInvitePage) => any): Promise<InteractionChannelInvitePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InteractionChannelInviteListInstance(version: V1, interactionSid: string, channelSid: string): InteractionChannelInviteListInstance;
interface InteractionChannelInvitePayload extends TwilioResponsePayload {
    invites: InteractionChannelInviteResource[];
}
interface InteractionChannelInviteResource {
    sid: string;
    interaction_sid: string;
    channel_sid: string;
    routing: any;
    url: string;
}
export declare class InteractionChannelInviteInstance {
    protected _version: V1;
    constructor(_version: V1, payload: InteractionChannelInviteResource, interactionSid: string, channelSid: string);
    /**
     * The unique string created by Twilio to identify an Interaction Channel Invite resource.
     */
    sid: string;
    /**
     * The Interaction SID for this Channel.
     */
    interactionSid: string;
    /**
     * The Channel SID for this Invite.
     */
    channelSid: string;
    /**
     * A JSON object representing the routing rules for the Interaction Channel. See [Outbound SMS Example](https://www.twilio.com/docs/flex/developer/conversations/interactions-api/interactions#agent-initiated-outbound-interactions) for an example Routing object. The Interactions resource uses TaskRouter for all routing functionality.   All attributes in the Routing object on your Interaction request body are added “as is” to the task. For a list of known attributes consumed by the Flex UI and/or Flex Insights, see [Known Task Attributes](https://www.twilio.com/docs/flex/developer/conversations/interactions-api#task-attributes).
     */
    routing: any;
    url: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        interactionSid: string;
        channelSid: string;
        routing: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class InteractionChannelInvitePage extends Page<V1, InteractionChannelInvitePayload, InteractionChannelInviteResource, InteractionChannelInviteInstance> {
    /**
     * Initialize the InteractionChannelInvitePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: InteractionChannelInviteSolution);
    /**
     * Build an instance of InteractionChannelInviteInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: InteractionChannelInviteResource): InteractionChannelInviteInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
