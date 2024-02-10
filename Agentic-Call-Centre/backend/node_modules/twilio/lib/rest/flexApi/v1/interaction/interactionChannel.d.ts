/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { InteractionChannelInviteListInstance } from "./interactionChannel/interactionChannelInvite";
import { InteractionChannelParticipantListInstance } from "./interactionChannel/interactionChannelParticipant";
export type InteractionChannelChannelStatus = "setup" | "active" | "failed" | "closed" | "inactive";
export type InteractionChannelType = "voice" | "sms" | "email" | "web" | "whatsapp" | "chat" | "messenger" | "gbm";
export type InteractionChannelUpdateChannelStatus = "closed" | "inactive";
/**
 * Options to pass to update a InteractionChannelInstance
 */
export interface InteractionChannelContextUpdateOptions {
    /**  */
    status: InteractionChannelUpdateChannelStatus;
    /** It changes the state of associated tasks. Routing status is required, When the channel status is set to `inactive`. Allowed Value for routing status is `closed`. Otherwise Optional, if not specified, all tasks will be set to `wrapping`. */
    routing?: any;
}
/**
 * Options to pass to each
 */
export interface InteractionChannelListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: InteractionChannelInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface InteractionChannelListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface InteractionChannelListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface InteractionChannelContext {
    invites: InteractionChannelInviteListInstance;
    participants: InteractionChannelParticipantListInstance;
    /**
     * Fetch a InteractionChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: InteractionChannelInstance) => any): Promise<InteractionChannelInstance>;
    /**
     * Update a InteractionChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionChannelInstance
     */
    update(params: InteractionChannelContextUpdateOptions, callback?: (error: Error | null, item?: InteractionChannelInstance) => any): Promise<InteractionChannelInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface InteractionChannelContextSolution {
    interactionSid: string;
    sid: string;
}
export declare class InteractionChannelContextImpl implements InteractionChannelContext {
    protected _version: V1;
    protected _solution: InteractionChannelContextSolution;
    protected _uri: string;
    protected _invites?: InteractionChannelInviteListInstance;
    protected _participants?: InteractionChannelParticipantListInstance;
    constructor(_version: V1, interactionSid: string, sid: string);
    get invites(): InteractionChannelInviteListInstance;
    get participants(): InteractionChannelParticipantListInstance;
    fetch(callback?: (error: Error | null, item?: InteractionChannelInstance) => any): Promise<InteractionChannelInstance>;
    update(params: InteractionChannelContextUpdateOptions, callback?: (error: Error | null, item?: InteractionChannelInstance) => any): Promise<InteractionChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): InteractionChannelContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface InteractionChannelPayload extends TwilioResponsePayload {
    channels: InteractionChannelResource[];
}
interface InteractionChannelResource {
    sid: string;
    interaction_sid: string;
    type: InteractionChannelType;
    status: InteractionChannelChannelStatus;
    error_code: number;
    error_message: string;
    url: string;
    links: Record<string, string>;
}
export declare class InteractionChannelInstance {
    protected _version: V1;
    protected _solution: InteractionChannelContextSolution;
    protected _context?: InteractionChannelContext;
    constructor(_version: V1, payload: InteractionChannelResource, interactionSid: string, sid?: string);
    /**
     * The unique string created by Twilio to identify an Interaction Channel resource, prefixed with UO.
     */
    sid: string;
    /**
     * The unique string created by Twilio to identify an Interaction resource, prefixed with KD.
     */
    interactionSid: string;
    type: InteractionChannelType;
    status: InteractionChannelChannelStatus;
    /**
     * The Twilio error code for a failed channel.
     */
    errorCode: number;
    /**
     * The error message for a failed channel.
     */
    errorMessage: string;
    url: string;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a InteractionChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: InteractionChannelInstance) => any): Promise<InteractionChannelInstance>;
    /**
     * Update a InteractionChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionChannelInstance
     */
    update(params: InteractionChannelContextUpdateOptions, callback?: (error: Error | null, item?: InteractionChannelInstance) => any): Promise<InteractionChannelInstance>;
    /**
     * Access the invites.
     */
    invites(): InteractionChannelInviteListInstance;
    /**
     * Access the participants.
     */
    participants(): InteractionChannelParticipantListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        interactionSid: string;
        type: InteractionChannelType;
        status: InteractionChannelChannelStatus;
        errorCode: number;
        errorMessage: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface InteractionChannelSolution {
    interactionSid: string;
}
export interface InteractionChannelListInstance {
    _version: V1;
    _solution: InteractionChannelSolution;
    _uri: string;
    (sid: string): InteractionChannelContext;
    get(sid: string): InteractionChannelContext;
    /**
     * Streams InteractionChannelInstance records from the API.
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
     * @param { InteractionChannelListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: InteractionChannelInstance, done: (err?: Error) => void) => void): void;
    each(params: InteractionChannelListInstanceEachOptions, callback?: (item: InteractionChannelInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of InteractionChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: InteractionChannelPage) => any): Promise<InteractionChannelPage>;
    /**
     * Lists InteractionChannelInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InteractionChannelListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: InteractionChannelInstance[]) => any): Promise<InteractionChannelInstance[]>;
    list(params: InteractionChannelListInstanceOptions, callback?: (error: Error | null, items: InteractionChannelInstance[]) => any): Promise<InteractionChannelInstance[]>;
    /**
     * Retrieve a single page of InteractionChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InteractionChannelListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: InteractionChannelPage) => any): Promise<InteractionChannelPage>;
    page(params: InteractionChannelListInstancePageOptions, callback?: (error: Error | null, items: InteractionChannelPage) => any): Promise<InteractionChannelPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InteractionChannelListInstance(version: V1, interactionSid: string): InteractionChannelListInstance;
export declare class InteractionChannelPage extends Page<V1, InteractionChannelPayload, InteractionChannelResource, InteractionChannelInstance> {
    /**
     * Initialize the InteractionChannelPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: InteractionChannelSolution);
    /**
     * Build an instance of InteractionChannelInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: InteractionChannelResource): InteractionChannelInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
