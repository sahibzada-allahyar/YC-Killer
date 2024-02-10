/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
export type EventLevel = "UNKNOWN" | "DEBUG" | "INFO" | "WARNING" | "ERROR";
export type EventTwilioEdge = "unknown_edge" | "carrier_edge" | "sip_edge" | "sdk_edge" | "client_edge";
/**
 * Options to pass to each
 */
export interface EventListInstanceEachOptions {
    /** The Edge of this Event. One of `unknown_edge`, `carrier_edge`, `sip_edge`, `sdk_edge` or `client_edge`. */
    edge?: EventTwilioEdge;
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
    /** The Edge of this Event. One of `unknown_edge`, `carrier_edge`, `sip_edge`, `sdk_edge` or `client_edge`. */
    edge?: EventTwilioEdge;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface EventListInstancePageOptions {
    /** The Edge of this Event. One of `unknown_edge`, `carrier_edge`, `sip_edge`, `sdk_edge` or `client_edge`. */
    edge?: EventTwilioEdge;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface EventSolution {
    callSid: string;
}
export interface EventListInstance {
    _version: V1;
    _solution: EventSolution;
    _uri: string;
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
export declare function EventListInstance(version: V1, callSid: string): EventListInstance;
interface EventPayload extends TwilioResponsePayload {
    events: EventResource[];
}
interface EventResource {
    timestamp: string;
    call_sid: string;
    account_sid: string;
    edge: EventTwilioEdge;
    group: string;
    level: EventLevel;
    name: string;
    carrier_edge: any;
    sip_edge: any;
    sdk_edge: any;
    client_edge: any;
}
export declare class EventInstance {
    protected _version: V1;
    constructor(_version: V1, payload: EventResource, callSid: string);
    /**
     * Event time.
     */
    timestamp: string;
    /**
     * The unique SID identifier of the Call.
     */
    callSid: string;
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    edge: EventTwilioEdge;
    /**
     * Event group.
     */
    group: string;
    level: EventLevel;
    /**
     * Event name.
     */
    name: string;
    /**
     * Represents the connection between Twilio and our immediate carrier partners. The events here describe the call lifecycle as reported by Twilio\'s carrier media gateways.
     */
    carrierEdge: any;
    /**
     * Represents the Twilio media gateway for SIP interface and SIP trunking calls. The events here describe the call lifecycle as reported by Twilio\'s public media gateways.
     */
    sipEdge: any;
    /**
     * Represents the Voice SDK running locally in the browser or in the Android/iOS application. The events here are emitted by the Voice SDK in response to certain call progress events, network changes, or call quality conditions.
     */
    sdkEdge: any;
    /**
     * Represents the Twilio media gateway for Client calls. The events here describe the call lifecycle as reported by Twilio\'s Voice SDK media gateways.
     */
    clientEdge: any;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        timestamp: string;
        callSid: string;
        accountSid: string;
        edge: EventTwilioEdge;
        group: string;
        level: EventLevel;
        name: string;
        carrierEdge: any;
        sipEdge: any;
        sdkEdge: any;
        clientEdge: any;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
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
