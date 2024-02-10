/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
export type InteractionChannelParticipantStatus = "closed" | "wrapup";
export type InteractionChannelParticipantType = "supervisor" | "customer" | "external" | "agent" | "unknown";
/**
 * Options to pass to update a InteractionChannelParticipantInstance
 */
export interface InteractionChannelParticipantContextUpdateOptions {
    /**  */
    status: InteractionChannelParticipantStatus;
}
/**
 * Options to pass to create a InteractionChannelParticipantInstance
 */
export interface InteractionChannelParticipantListInstanceCreateOptions {
    /**  */
    type: InteractionChannelParticipantType;
    /** JSON representing the Media Properties for the new Participant. */
    mediaProperties: any;
    /** Object representing the Routing Properties for the new Participant. */
    routingProperties?: any;
}
/**
 * Options to pass to each
 */
export interface InteractionChannelParticipantListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: InteractionChannelParticipantInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface InteractionChannelParticipantListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface InteractionChannelParticipantListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface InteractionChannelParticipantContext {
    /**
     * Update a InteractionChannelParticipantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionChannelParticipantInstance
     */
    update(params: InteractionChannelParticipantContextUpdateOptions, callback?: (error: Error | null, item?: InteractionChannelParticipantInstance) => any): Promise<InteractionChannelParticipantInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface InteractionChannelParticipantContextSolution {
    interactionSid: string;
    channelSid: string;
    sid: string;
}
export declare class InteractionChannelParticipantContextImpl implements InteractionChannelParticipantContext {
    protected _version: V1;
    protected _solution: InteractionChannelParticipantContextSolution;
    protected _uri: string;
    constructor(_version: V1, interactionSid: string, channelSid: string, sid: string);
    update(params: InteractionChannelParticipantContextUpdateOptions, callback?: (error: Error | null, item?: InteractionChannelParticipantInstance) => any): Promise<InteractionChannelParticipantInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): InteractionChannelParticipantContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface InteractionChannelParticipantPayload extends TwilioResponsePayload {
    participants: InteractionChannelParticipantResource[];
}
interface InteractionChannelParticipantResource {
    sid: string;
    type: InteractionChannelParticipantType;
    interaction_sid: string;
    channel_sid: string;
    url: string;
    routing_properties: any;
}
export declare class InteractionChannelParticipantInstance {
    protected _version: V1;
    protected _solution: InteractionChannelParticipantContextSolution;
    protected _context?: InteractionChannelParticipantContext;
    constructor(_version: V1, payload: InteractionChannelParticipantResource, interactionSid: string, channelSid: string, sid?: string);
    /**
     * The unique string created by Twilio to identify an Interaction Channel Participant resource.
     */
    sid: string;
    type: InteractionChannelParticipantType;
    /**
     * The Interaction Sid for this channel.
     */
    interactionSid: string;
    /**
     * The Channel Sid for this Participant.
     */
    channelSid: string;
    url: string;
    /**
     * The Participant\'s routing properties.
     */
    routingProperties: any;
    private get _proxy();
    /**
     * Update a InteractionChannelParticipantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionChannelParticipantInstance
     */
    update(params: InteractionChannelParticipantContextUpdateOptions, callback?: (error: Error | null, item?: InteractionChannelParticipantInstance) => any): Promise<InteractionChannelParticipantInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        type: InteractionChannelParticipantType;
        interactionSid: string;
        channelSid: string;
        url: string;
        routingProperties: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface InteractionChannelParticipantSolution {
    interactionSid: string;
    channelSid: string;
}
export interface InteractionChannelParticipantListInstance {
    _version: V1;
    _solution: InteractionChannelParticipantSolution;
    _uri: string;
    (sid: string): InteractionChannelParticipantContext;
    get(sid: string): InteractionChannelParticipantContext;
    /**
     * Create a InteractionChannelParticipantInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InteractionChannelParticipantInstance
     */
    create(params: InteractionChannelParticipantListInstanceCreateOptions, callback?: (error: Error | null, item?: InteractionChannelParticipantInstance) => any): Promise<InteractionChannelParticipantInstance>;
    /**
     * Streams InteractionChannelParticipantInstance records from the API.
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
     * @param { InteractionChannelParticipantListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: InteractionChannelParticipantInstance, done: (err?: Error) => void) => void): void;
    each(params: InteractionChannelParticipantListInstanceEachOptions, callback?: (item: InteractionChannelParticipantInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of InteractionChannelParticipantInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: InteractionChannelParticipantPage) => any): Promise<InteractionChannelParticipantPage>;
    /**
     * Lists InteractionChannelParticipantInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InteractionChannelParticipantListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: InteractionChannelParticipantInstance[]) => any): Promise<InteractionChannelParticipantInstance[]>;
    list(params: InteractionChannelParticipantListInstanceOptions, callback?: (error: Error | null, items: InteractionChannelParticipantInstance[]) => any): Promise<InteractionChannelParticipantInstance[]>;
    /**
     * Retrieve a single page of InteractionChannelParticipantInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InteractionChannelParticipantListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: InteractionChannelParticipantPage) => any): Promise<InteractionChannelParticipantPage>;
    page(params: InteractionChannelParticipantListInstancePageOptions, callback?: (error: Error | null, items: InteractionChannelParticipantPage) => any): Promise<InteractionChannelParticipantPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InteractionChannelParticipantListInstance(version: V1, interactionSid: string, channelSid: string): InteractionChannelParticipantListInstance;
export declare class InteractionChannelParticipantPage extends Page<V1, InteractionChannelParticipantPayload, InteractionChannelParticipantResource, InteractionChannelParticipantInstance> {
    /**
     * Initialize the InteractionChannelParticipantPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: InteractionChannelParticipantSolution);
    /**
     * Build an instance of InteractionChannelParticipantInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: InteractionChannelParticipantResource): InteractionChannelParticipantInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
