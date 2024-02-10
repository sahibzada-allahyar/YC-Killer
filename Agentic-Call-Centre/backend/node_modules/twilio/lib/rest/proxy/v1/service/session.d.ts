/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { InteractionListInstance } from "./session/interaction";
import { ParticipantListInstance } from "./session/participant";
export type SessionMode = "message-only" | "voice-only" | "voice-and-message";
export type SessionStatus = "open" | "in-progress" | "closed" | "failed" | "unknown";
/**
 * Options to pass to update a SessionInstance
 */
export interface SessionContextUpdateOptions {
    /** The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date when the Session should expire. If this is value is present, it overrides the `ttl` value. */
    dateExpiry?: Date;
    /** The time, in seconds, when the session will expire. The time is measured from the last Session create or the Session\\\'s last Interaction. */
    ttl?: number;
    /**  */
    status?: SessionStatus;
}
/**
 * Options to pass to create a SessionInstance
 */
export interface SessionListInstanceCreateOptions {
    /** An application-defined string that uniquely identifies the resource. This value must be 191 characters or fewer in length and be unique. **This value should not have PII.** */
    uniqueName?: string;
    /** The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date when the Session should expire. If this is value is present, it overrides the `ttl` value. */
    dateExpiry?: Date;
    /** The time, in seconds, when the session will expire. The time is measured from the last Session create or the Session\\\'s last Interaction. */
    ttl?: number;
    /**  */
    mode?: SessionMode;
    /**  */
    status?: SessionStatus;
    /** The Participant objects to include in the new session. */
    participants?: Array<any>;
}
/**
 * Options to pass to each
 */
export interface SessionListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: SessionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface SessionListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface SessionListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface SessionContext {
    interactions: InteractionListInstance;
    participants: ParticipantListInstance;
    /**
     * Remove a SessionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SessionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SessionInstance
     */
    fetch(callback?: (error: Error | null, item?: SessionInstance) => any): Promise<SessionInstance>;
    /**
     * Update a SessionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SessionInstance
     */
    update(callback?: (error: Error | null, item?: SessionInstance) => any): Promise<SessionInstance>;
    /**
     * Update a SessionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SessionInstance
     */
    update(params: SessionContextUpdateOptions, callback?: (error: Error | null, item?: SessionInstance) => any): Promise<SessionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface SessionContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class SessionContextImpl implements SessionContext {
    protected _version: V1;
    protected _solution: SessionContextSolution;
    protected _uri: string;
    protected _interactions?: InteractionListInstance;
    protected _participants?: ParticipantListInstance;
    constructor(_version: V1, serviceSid: string, sid: string);
    get interactions(): InteractionListInstance;
    get participants(): ParticipantListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: SessionInstance) => any): Promise<SessionInstance>;
    update(params?: SessionContextUpdateOptions | ((error: Error | null, item?: SessionInstance) => any), callback?: (error: Error | null, item?: SessionInstance) => any): Promise<SessionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SessionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SessionPayload extends TwilioResponsePayload {
    sessions: SessionResource[];
}
interface SessionResource {
    sid: string;
    service_sid: string;
    account_sid: string;
    date_started: Date;
    date_ended: Date;
    date_last_interaction: Date;
    date_expiry: Date;
    unique_name: string;
    status: SessionStatus;
    closed_reason: string;
    ttl: number;
    mode: SessionMode;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class SessionInstance {
    protected _version: V1;
    protected _solution: SessionContextSolution;
    protected _context?: SessionContext;
    constructor(_version: V1, payload: SessionResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Session resource.
     */
    sid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/proxy/api/service) the session is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Session resource.
     */
    accountSid: string;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date when the Session started.
     */
    dateStarted: Date;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date when the Session ended.
     */
    dateEnded: Date;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date when the Session last had an interaction.
     */
    dateLastInteraction: Date;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date when the Session should expire. If this is value is present, it overrides the `ttl` value.
     */
    dateExpiry: Date;
    /**
     * An application-defined string that uniquely identifies the resource. This value must be 191 characters or fewer in length and be unique. Supports UTF-8 characters. **This value should not have PII.**
     */
    uniqueName: string;
    status: SessionStatus;
    /**
     * The reason the Session ended.
     */
    closedReason: string;
    /**
     * The time, in seconds, when the session will expire. The time is measured from the last Session create or the Session\'s last Interaction.
     */
    ttl: number;
    mode: SessionMode;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the resource was created.
     */
    dateCreated: Date;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the resource was last updated.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Session resource.
     */
    url: string;
    /**
     * The URLs of resources related to the Session.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a SessionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a SessionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SessionInstance
     */
    fetch(callback?: (error: Error | null, item?: SessionInstance) => any): Promise<SessionInstance>;
    /**
     * Update a SessionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SessionInstance
     */
    update(callback?: (error: Error | null, item?: SessionInstance) => any): Promise<SessionInstance>;
    /**
     * Update a SessionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SessionInstance
     */
    update(params: SessionContextUpdateOptions, callback?: (error: Error | null, item?: SessionInstance) => any): Promise<SessionInstance>;
    /**
     * Access the interactions.
     */
    interactions(): InteractionListInstance;
    /**
     * Access the participants.
     */
    participants(): ParticipantListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        serviceSid: string;
        accountSid: string;
        dateStarted: Date;
        dateEnded: Date;
        dateLastInteraction: Date;
        dateExpiry: Date;
        uniqueName: string;
        status: SessionStatus;
        closedReason: string;
        ttl: number;
        mode: SessionMode;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SessionSolution {
    serviceSid: string;
}
export interface SessionListInstance {
    _version: V1;
    _solution: SessionSolution;
    _uri: string;
    (sid: string): SessionContext;
    get(sid: string): SessionContext;
    /**
     * Create a SessionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SessionInstance
     */
    create(callback?: (error: Error | null, item?: SessionInstance) => any): Promise<SessionInstance>;
    /**
     * Create a SessionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SessionInstance
     */
    create(params: SessionListInstanceCreateOptions, callback?: (error: Error | null, item?: SessionInstance) => any): Promise<SessionInstance>;
    /**
     * Streams SessionInstance records from the API.
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
     * @param { SessionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: SessionInstance, done: (err?: Error) => void) => void): void;
    each(params: SessionListInstanceEachOptions, callback?: (item: SessionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of SessionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: SessionPage) => any): Promise<SessionPage>;
    /**
     * Lists SessionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SessionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: SessionInstance[]) => any): Promise<SessionInstance[]>;
    list(params: SessionListInstanceOptions, callback?: (error: Error | null, items: SessionInstance[]) => any): Promise<SessionInstance[]>;
    /**
     * Retrieve a single page of SessionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { SessionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: SessionPage) => any): Promise<SessionPage>;
    page(params: SessionListInstancePageOptions, callback?: (error: Error | null, items: SessionPage) => any): Promise<SessionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SessionListInstance(version: V1, serviceSid: string): SessionListInstance;
export declare class SessionPage extends Page<V1, SessionPayload, SessionResource, SessionInstance> {
    /**
     * Initialize the SessionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: SessionSolution);
    /**
     * Build an instance of SessionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: SessionResource): SessionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
