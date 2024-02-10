/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to each
 */
export interface EventListInstanceEachOptions {
    /** Only include Events that occurred on or before this date, specified in GMT as an [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time. */
    endDate?: Date;
    /** The type of Events to read. Returns only Events of the type specified. */
    eventType?: string;
    /** The period of events to read in minutes. Returns only Events that occurred since this many minutes in the past. The default is `15` minutes. Task Attributes for Events occuring more 43,200 minutes ago will be redacted. */
    minutes?: number;
    /** The SID of the Reservation with the Events to read. Returns only Events that pertain to the specified Reservation. */
    reservationSid?: string;
    /** Only include Events from on or after this date and time, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. Task Attributes for Events older than 30 days will be redacted. */
    startDate?: Date;
    /** The SID of the TaskQueue with the Events to read. Returns only the Events that pertain to the specified TaskQueue. */
    taskQueueSid?: string;
    /** The SID of the Task with the Events to read. Returns only the Events that pertain to the specified Task. */
    taskSid?: string;
    /** The SID of the Worker with the Events to read. Returns only the Events that pertain to the specified Worker. */
    workerSid?: string;
    /** The SID of the Workflow with the Events to read. Returns only the Events that pertain to the specified Workflow. */
    workflowSid?: string;
    /** The TaskChannel with the Events to read. Returns only the Events that pertain to the specified TaskChannel. */
    taskChannel?: string;
    /** The SID of the Event resource to read. */
    sid?: string;
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
    /** Only include Events that occurred on or before this date, specified in GMT as an [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time. */
    endDate?: Date;
    /** The type of Events to read. Returns only Events of the type specified. */
    eventType?: string;
    /** The period of events to read in minutes. Returns only Events that occurred since this many minutes in the past. The default is `15` minutes. Task Attributes for Events occuring more 43,200 minutes ago will be redacted. */
    minutes?: number;
    /** The SID of the Reservation with the Events to read. Returns only Events that pertain to the specified Reservation. */
    reservationSid?: string;
    /** Only include Events from on or after this date and time, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. Task Attributes for Events older than 30 days will be redacted. */
    startDate?: Date;
    /** The SID of the TaskQueue with the Events to read. Returns only the Events that pertain to the specified TaskQueue. */
    taskQueueSid?: string;
    /** The SID of the Task with the Events to read. Returns only the Events that pertain to the specified Task. */
    taskSid?: string;
    /** The SID of the Worker with the Events to read. Returns only the Events that pertain to the specified Worker. */
    workerSid?: string;
    /** The SID of the Workflow with the Events to read. Returns only the Events that pertain to the specified Workflow. */
    workflowSid?: string;
    /** The TaskChannel with the Events to read. Returns only the Events that pertain to the specified TaskChannel. */
    taskChannel?: string;
    /** The SID of the Event resource to read. */
    sid?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface EventListInstancePageOptions {
    /** Only include Events that occurred on or before this date, specified in GMT as an [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date-time. */
    endDate?: Date;
    /** The type of Events to read. Returns only Events of the type specified. */
    eventType?: string;
    /** The period of events to read in minutes. Returns only Events that occurred since this many minutes in the past. The default is `15` minutes. Task Attributes for Events occuring more 43,200 minutes ago will be redacted. */
    minutes?: number;
    /** The SID of the Reservation with the Events to read. Returns only Events that pertain to the specified Reservation. */
    reservationSid?: string;
    /** Only include Events from on or after this date and time, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format. Task Attributes for Events older than 30 days will be redacted. */
    startDate?: Date;
    /** The SID of the TaskQueue with the Events to read. Returns only the Events that pertain to the specified TaskQueue. */
    taskQueueSid?: string;
    /** The SID of the Task with the Events to read. Returns only the Events that pertain to the specified Task. */
    taskSid?: string;
    /** The SID of the Worker with the Events to read. Returns only the Events that pertain to the specified Worker. */
    workerSid?: string;
    /** The SID of the Workflow with the Events to read. Returns only the Events that pertain to the specified Workflow. */
    workflowSid?: string;
    /** The TaskChannel with the Events to read. Returns only the Events that pertain to the specified TaskChannel. */
    taskChannel?: string;
    /** The SID of the Event resource to read. */
    sid?: string;
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
    workspaceSid: string;
    sid: string;
}
export declare class EventContextImpl implements EventContext {
    protected _version: V1;
    protected _solution: EventContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string, sid: string);
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
    actor_url: string;
    description: string;
    event_data: any;
    event_date: Date;
    event_date_ms: number;
    event_type: string;
    resource_sid: string;
    resource_type: string;
    resource_url: string;
    sid: string;
    source: string;
    source_ip_address: string;
    url: string;
    workspace_sid: string;
}
export declare class EventInstance {
    protected _version: V1;
    protected _solution: EventContextSolution;
    protected _context?: EventContext;
    constructor(_version: V1, payload: EventResource, workspaceSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Event resource.
     */
    accountSid: string;
    /**
     * The SID of the resource that triggered the event.
     */
    actorSid: string;
    /**
     * The type of resource that triggered the event.
     */
    actorType: string;
    /**
     * The absolute URL of the resource that triggered the event.
     */
    actorUrl: string;
    /**
     * A description of the event.
     */
    description: string;
    /**
     * Data about the event. For more information, see [Event types](https://www.twilio.com/docs/taskrouter/api/event#event-types).
     */
    eventData: any;
    /**
     * The time the event was sent, specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    eventDate: Date;
    /**
     * The time the event was sent in milliseconds.
     */
    eventDateMs: number;
    /**
     * The identifier for the event.
     */
    eventType: string;
    /**
     * The SID of the object the event is most relevant to, such as a TaskSid, ReservationSid, or a  WorkerSid.
     */
    resourceSid: string;
    /**
     * The type of object the event is most relevant to, such as a Task, Reservation, or a Worker).
     */
    resourceType: string;
    /**
     * The URL of the resource the event is most relevant to.
     */
    resourceUrl: string;
    /**
     * The unique string that we created to identify the Event resource.
     */
    sid: string;
    /**
     * Where the Event originated.
     */
    source: string;
    /**
     * The IP from which the Event originated.
     */
    sourceIpAddress: string;
    /**
     * The absolute URL of the Event resource.
     */
    url: string;
    /**
     * The SID of the Workspace that contains the Event.
     */
    workspaceSid: string;
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
        actorUrl: string;
        description: string;
        eventData: any;
        eventDate: Date;
        eventDateMs: number;
        eventType: string;
        resourceSid: string;
        resourceType: string;
        resourceUrl: string;
        sid: string;
        source: string;
        sourceIpAddress: string;
        url: string;
        workspaceSid: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface EventSolution {
    workspaceSid: string;
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
export declare function EventListInstance(version: V1, workspaceSid: string): EventListInstance;
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
