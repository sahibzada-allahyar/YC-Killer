/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to each
 */
export interface InsightsSegmentsListInstanceEachOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** To unique id of the segment */
    segmentId?: string;
    /** The list of reservation Ids */
    reservationId?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: InsightsSegmentsInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface InsightsSegmentsListInstanceOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** To unique id of the segment */
    segmentId?: string;
    /** The list of reservation Ids */
    reservationId?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface InsightsSegmentsListInstancePageOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** To unique id of the segment */
    segmentId?: string;
    /** The list of reservation Ids */
    reservationId?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface InsightsSegmentsSolution {
}
export interface InsightsSegmentsListInstance {
    _version: V1;
    _solution: InsightsSegmentsSolution;
    _uri: string;
    /**
     * Streams InsightsSegmentsInstance records from the API.
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
     * @param { InsightsSegmentsListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: InsightsSegmentsInstance, done: (err?: Error) => void) => void): void;
    each(params: InsightsSegmentsListInstanceEachOptions, callback?: (item: InsightsSegmentsInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of InsightsSegmentsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: InsightsSegmentsPage) => any): Promise<InsightsSegmentsPage>;
    /**
     * Lists InsightsSegmentsInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InsightsSegmentsListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: InsightsSegmentsInstance[]) => any): Promise<InsightsSegmentsInstance[]>;
    list(params: InsightsSegmentsListInstanceOptions, callback?: (error: Error | null, items: InsightsSegmentsInstance[]) => any): Promise<InsightsSegmentsInstance[]>;
    /**
     * Retrieve a single page of InsightsSegmentsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InsightsSegmentsListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: InsightsSegmentsPage) => any): Promise<InsightsSegmentsPage>;
    page(params: InsightsSegmentsListInstancePageOptions, callback?: (error: Error | null, items: InsightsSegmentsPage) => any): Promise<InsightsSegmentsPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InsightsSegmentsListInstance(version: V1): InsightsSegmentsListInstance;
interface InsightsSegmentsPayload extends TwilioResponsePayload {
    segments: InsightsSegmentsResource[];
}
interface InsightsSegmentsResource {
    segment_id: string;
    external_id: string;
    queue: string;
    external_contact: string;
    external_segment_link_id: string;
    date: string;
    account_id: string;
    external_segment_link: string;
    agent_id: string;
    agent_phone: string;
    agent_name: string;
    agent_team_name: string;
    agent_team_name_in_hierarchy: string;
    agent_link: string;
    customer_phone: string;
    customer_name: string;
    customer_link: string;
    segment_recording_offset: string;
    media: any;
    assessment_type: any;
    assessment_percentage: any;
    url: string;
}
export declare class InsightsSegmentsInstance {
    protected _version: V1;
    constructor(_version: V1, payload: InsightsSegmentsResource);
    /**
     * To unique id of the segment
     */
    segmentId: string;
    /**
     * The unique id for the conversation.
     */
    externalId: string;
    queue: string;
    externalContact: string;
    /**
     * The uuid for the external_segment_link.
     */
    externalSegmentLinkId: string;
    /**
     * The date of the conversation.
     */
    date: string;
    /**
     * The unique id for the account.
     */
    accountId: string;
    /**
     * The hyperlink to recording of the task event.
     */
    externalSegmentLink: string;
    /**
     * The unique id for the agent.
     */
    agentId: string;
    /**
     * The phone number of the agent.
     */
    agentPhone: string;
    /**
     * The name of the agent.
     */
    agentName: string;
    /**
     * The team name to which agent belongs.
     */
    agentTeamName: string;
    /**
     * he team name to which agent belongs.
     */
    agentTeamNameInHierarchy: string;
    /**
     * The link to the agent conversation.
     */
    agentLink: string;
    /**
     * The phone number of the customer.
     */
    customerPhone: string;
    /**
     * The name of the customer.
     */
    customerName: string;
    /**
     * The link to the customer conversation.
     */
    customerLink: string;
    /**
     * The offset value for the recording.
     */
    segmentRecordingOffset: string;
    /**
     * The media identifiers of the conversation.
     */
    media: any;
    /**
     * The type of the assessment.
     */
    assessmentType: any;
    /**
     * The percentage scored on the Assessments.
     */
    assessmentPercentage: any;
    url: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        segmentId: string;
        externalId: string;
        queue: string;
        externalContact: string;
        externalSegmentLinkId: string;
        date: string;
        accountId: string;
        externalSegmentLink: string;
        agentId: string;
        agentPhone: string;
        agentName: string;
        agentTeamName: string;
        agentTeamNameInHierarchy: string;
        agentLink: string;
        customerPhone: string;
        customerName: string;
        customerLink: string;
        segmentRecordingOffset: string;
        media: any;
        assessmentType: any;
        assessmentPercentage: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class InsightsSegmentsPage extends Page<V1, InsightsSegmentsPayload, InsightsSegmentsResource, InsightsSegmentsInstance> {
    /**
     * Initialize the InsightsSegmentsPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: InsightsSegmentsSolution);
    /**
     * Build an instance of InsightsSegmentsInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: InsightsSegmentsResource): InsightsSegmentsInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
