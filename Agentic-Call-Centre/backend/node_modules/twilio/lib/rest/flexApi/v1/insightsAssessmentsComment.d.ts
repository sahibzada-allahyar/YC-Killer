/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to create a InsightsAssessmentsCommentInstance
 */
export interface InsightsAssessmentsCommentListInstanceCreateOptions {
    /** The ID of the category */
    categoryId: string;
    /** The name of the category */
    categoryName: string;
    /** The Assessment comment. */
    comment: string;
    /** The id of the segment. */
    segmentId: string;
    /** The id of the agent. */
    agentId: string;
    /** The offset */
    offset: number;
    /** The Authorization HTTP request header */
    authorization?: string;
}
/**
 * Options to pass to each
 */
export interface InsightsAssessmentsCommentListInstanceEachOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The id of the segment. */
    segmentId?: string;
    /** The id of the agent. */
    agentId?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: InsightsAssessmentsCommentInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface InsightsAssessmentsCommentListInstanceOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The id of the segment. */
    segmentId?: string;
    /** The id of the agent. */
    agentId?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface InsightsAssessmentsCommentListInstancePageOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The id of the segment. */
    segmentId?: string;
    /** The id of the agent. */
    agentId?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface InsightsAssessmentsCommentSolution {
}
export interface InsightsAssessmentsCommentListInstance {
    _version: V1;
    _solution: InsightsAssessmentsCommentSolution;
    _uri: string;
    /**
     * Create a InsightsAssessmentsCommentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsAssessmentsCommentInstance
     */
    create(params: InsightsAssessmentsCommentListInstanceCreateOptions, callback?: (error: Error | null, item?: InsightsAssessmentsCommentInstance) => any): Promise<InsightsAssessmentsCommentInstance>;
    /**
     * Streams InsightsAssessmentsCommentInstance records from the API.
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
     * @param { InsightsAssessmentsCommentListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: InsightsAssessmentsCommentInstance, done: (err?: Error) => void) => void): void;
    each(params: InsightsAssessmentsCommentListInstanceEachOptions, callback?: (item: InsightsAssessmentsCommentInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of InsightsAssessmentsCommentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: InsightsAssessmentsCommentPage) => any): Promise<InsightsAssessmentsCommentPage>;
    /**
     * Lists InsightsAssessmentsCommentInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InsightsAssessmentsCommentListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: InsightsAssessmentsCommentInstance[]) => any): Promise<InsightsAssessmentsCommentInstance[]>;
    list(params: InsightsAssessmentsCommentListInstanceOptions, callback?: (error: Error | null, items: InsightsAssessmentsCommentInstance[]) => any): Promise<InsightsAssessmentsCommentInstance[]>;
    /**
     * Retrieve a single page of InsightsAssessmentsCommentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InsightsAssessmentsCommentListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: InsightsAssessmentsCommentPage) => any): Promise<InsightsAssessmentsCommentPage>;
    page(params: InsightsAssessmentsCommentListInstancePageOptions, callback?: (error: Error | null, items: InsightsAssessmentsCommentPage) => any): Promise<InsightsAssessmentsCommentPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InsightsAssessmentsCommentListInstance(version: V1): InsightsAssessmentsCommentListInstance;
interface InsightsAssessmentsCommentPayload extends TwilioResponsePayload {
    comments: InsightsAssessmentsCommentResource[];
}
interface InsightsAssessmentsCommentResource {
    account_sid: string;
    assessment_sid: string;
    comment: any;
    offset: number;
    report: boolean;
    weight: number;
    agent_id: string;
    segment_id: string;
    user_name: string;
    user_email: string;
    timestamp: number;
    url: string;
}
export declare class InsightsAssessmentsCommentInstance {
    protected _version: V1;
    constructor(_version: V1, payload: InsightsAssessmentsCommentResource);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Flex Insights resource and owns this resource.
     */
    accountSid: string;
    /**
     * The SID of the assessment.
     */
    assessmentSid: string;
    /**
     * The comment added for assessment.
     */
    comment: any;
    /**
     * The offset
     */
    offset: number;
    /**
     * The flag indicating if this assessment is part of report
     */
    report: boolean;
    /**
     * The weightage given to this comment
     */
    weight: number;
    /**
     * The id of the agent.
     */
    agentId: string;
    /**
     * The id of the segment.
     */
    segmentId: string;
    /**
     * The name of the user.
     */
    userName: string;
    /**
     * The email id of the user.
     */
    userEmail: string;
    /**
     * The timestamp when the record is inserted
     */
    timestamp: number;
    url: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        assessmentSid: string;
        comment: any;
        offset: number;
        report: boolean;
        weight: number;
        agentId: string;
        segmentId: string;
        userName: string;
        userEmail: string;
        timestamp: number;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class InsightsAssessmentsCommentPage extends Page<V1, InsightsAssessmentsCommentPayload, InsightsAssessmentsCommentResource, InsightsAssessmentsCommentInstance> {
    /**
     * Initialize the InsightsAssessmentsCommentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: InsightsAssessmentsCommentSolution);
    /**
     * Build an instance of InsightsAssessmentsCommentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: InsightsAssessmentsCommentResource): InsightsAssessmentsCommentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
