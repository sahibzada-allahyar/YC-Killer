/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to each
 */
export interface EventListInstanceEachOptions {
    /** Only include events initiated by this Actor. Useful for auditing actions taken by specific users or API credentials. */
    actorSid?: string;
    /** Only include events of this [Event Type](https://www.twilio.com/docs/usage/monitor-events#event-types). */
    eventType?: string;
    /** Only include events that refer to this resource. Useful for discovering the history of a specific resource. */
    resourceSid?: string;
    /** Only include events that originated from this IP address. Useful for tracking suspicious activity originating from the API or the Twilio Console. */
    sourceIpAddress?: string;
    /** Only include events that occurred on or after this date. Specify the date in GMT and [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only include events that occurred on or before this date. Specify the date in GMT and [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    endDate?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: EventInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface EventListInstanceOptions {
    /** Only include events initiated by this Actor. Useful for auditing actions taken by specific users or API credentials. */
    actorSid?: string;
    /** Only include events of this [Event Type](https://www.twilio.com/docs/usage/monitor-events#event-types). */
    eventType?: string;
    /** Only include events that refer to this resource. Useful for discovering the history of a specific resource. */
    resourceSid?: string;
    /** Only include events that originated from this IP address. Useful for tracking suspicious activity originating from the API or the Twilio Console. */
    sourceIpAddress?: string;
    /** Only include events that occurred on or after this date. Specify the date in GMT and [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only include events that occurred on or before this date. Specify the date in GMT and [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    endDate?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface EventListInstancePageOptions {
    /** Only include events initiated by this Actor. Useful for auditing actions taken by specific users or API credentials. */
    actorSid?: string;
    /** Only include events of this [Event Type](https://www.twilio.com/docs/usage/monitor-events#event-types). */
    eventType?: string;
    /** Only include events that refer to this resource. Useful for discovering the history of a specific resource. */
    resourceSid?: string;
    /** Only include events that originated from this IP address. Useful for tracking suspicious activity originating from the API or the Twilio Console. */
    sourceIpAddress?: string;
    /** Only include events that occurred on or after this date. Specify the date in GMT and [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    startDate?: Date;
    /** Only include events that occurred on or before this date. Specify the date in GMT and [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. */
    endDate?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface EventContext {
    /**
     * Fetch a EventInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EventInstance
     */
    fetch(callback?: (error: Error | null, item?: EventInstance) => any): Promise<EventInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface EventContextSolution {
    sid: string;
}
export declare class EventContextImpl implements EventContext {
    protected _version: V1;
    protected _solution: EventContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    fetch(callback?: (error: Error | null, item?: EventInstance) => any): Promise<EventInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): EventContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface EventPayload extends TwilioResponsePayload {
    events: EventResource[];
}
interface EventResource {
    account_sid: string;
    actor_sid: string;
    actor_type: string;
    description: string;
    event_data: any;
    event_date: Date;
    event_type: string;
    resource_sid: string;
    resource_type: string;
    sid: string;
    source: string;
    source_ip_address: string;
    url: string;
    links: Record<string, string>;
}
export declare class EventInstance {
    protected _version: V1;
    protected _solution: EventContextSolution;
    protected _context?: EventContext;
    constructor(_version: V1, payload: EventResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Event resource.
     */
    accountSid: string;
    /**
     * The SID of the actor that caused the event, if available. Can be `null`.
     */
    actorSid: string;
    /**
     * The type of actor that caused the event. Can be: `user` for a change made by a logged-in user in the Twilio Console, `account` for an event caused by an API request by an authenticating Account, `twilio-admin` for an event caused by a Twilio employee, and so on.
     */
    actorType: string;
    /**
     * A description of the event. Can be `null`.
     */
    description: string;
    /**
     * An object with additional data about the event. The  contents depend on `event_type`. For example, event-types of the form `RESOURCE.updated`, this value contains a `resource_properties` dictionary that describes the previous and updated properties of the resource.
     */
    eventData: any;
    /**
     * The date and time in GMT when the event was recorded specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    eventDate: Date;
    /**
     * The event\'s type. Event-types are typically in the form: `RESOURCE_TYPE.ACTION`, where `RESOURCE_TYPE` is the type of resource that was affected and `ACTION` is what happened to it. For example, `phone-number.created`. For a full list of all event-types, see the [Monitor Event Types](https://www.twilio.com/docs/usage/monitor-events#event-types).
     */
    eventType: string;
    /**
     * The SID of the resource that was affected.
     */
    resourceSid: string;
    /**
     * The type of resource that was affected. For a full list of all resource-types, see the [Monitor Event Types](https://www.twilio.com/docs/usage/monitor-events#event-types).
     */
    resourceType: string;
    /**
     * The unique string that we created to identify the Event resource.
     */
    sid: string;
    /**
     * The originating system or interface that caused the event.  Can be: `web` for events caused by user action in the Twilio Console, `api` for events caused by a request to our API, or   `twilio` for events caused by an automated or internal Twilio system.
     */
    source: string;
    /**
     * The IP address of the source, if the source is outside the Twilio cloud. This value is `null` for events with `source` of `twilio`
     */
    sourceIpAddress: string;
    /**
     * The absolute URL of the resource that was affected. Can be `null`.
     */
    url: string;
    /**
     * The absolute URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a EventInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EventInstance
     */
    fetch(callback?: (error: Error | null, item?: EventInstance) => any): Promise<EventInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        actorSid: string;
        actorType: string;
        description: string;
        eventData: any;
        eventDate: Date;
        eventType: string;
        resourceSid: string;
        resourceType: string;
        sid: string;
        source: string;
        sourceIpAddress: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface EventSolution {
}
export interface EventListInstance {
    _version: V1;
    _solution: EventSolution;
    _uri: string;
    (sid: string): EventContext;
    get(sid: string): EventContext;
    /**
     * Streams EventInstance records from the API.
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
     * @param { EventListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: EventInstance, done: (err?: Error) => void) => void): void;
    each(params: EventListInstanceEachOptions, callback?: (item: EventInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of EventInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: EventPage) => any): Promise<EventPage>;
    /**
     * Lists EventInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EventListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: EventInstance[]) => any): Promise<EventInstance[]>;
    list(params: EventListInstanceOptions, callback?: (error: Error | null, items: EventInstance[]) => any): Promise<EventInstance[]>;
    /**
     * Retrieve a single page of EventInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EventListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: EventPage) => any): Promise<EventPage>;
    page(params: EventListInstancePageOptions, callback?: (error: Error | null, items: EventPage) => any): Promise<EventPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function EventListInstance(version: V1): EventListInstance;
export declare class EventPage extends Page<V1, EventPayload, EventResource, EventInstance> {
    /**
     * Initialize the EventPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: EventSolution);
    /**
     * Build an instance of EventInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: EventResource): EventInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
