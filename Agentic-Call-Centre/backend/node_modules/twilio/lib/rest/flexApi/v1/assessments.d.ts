/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to update a AssessmentsInstance
 */
export interface AssessmentsContextUpdateOptions {
    /** The offset of the conversation */
    offset: number;
    /** The answer text selected by user */
    answerText: string;
    /** The id of the answer selected by user */
    answerId: string;
    /** The Authorization HTTP request header */
    authorization?: string;
}
/**
 * Options to pass to create a AssessmentsInstance
 */
export interface AssessmentsListInstanceCreateOptions {
    /** The SID of the category  */
    categorySid: string;
    /** The name of the category */
    categoryName: string;
    /** Segment Id of the conversation */
    segmentId: string;
    /** The id of the Agent */
    agentId: string;
    /** The offset of the conversation. */
    offset: number;
    /** The question SID selected for assessment */
    metricId: string;
    /** The question name of the assessment */
    metricName: string;
    /** The answer text selected by user */
    answerText: string;
    /** The id of the answer selected by user */
    answerId: string;
    /** Questionnaire SID of the associated question */
    questionnaireSid: string;
    /** The Authorization HTTP request header */
    authorization?: string;
}
/**
 * Options to pass to each
 */
export interface AssessmentsListInstanceEachOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The id of the segment. */
    segmentId?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: AssessmentsInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface AssessmentsListInstanceOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The id of the segment. */
    segmentId?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface AssessmentsListInstancePageOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The id of the segment. */
    segmentId?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface AssessmentsContext {
    /**
     * Update a AssessmentsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssessmentsInstance
     */
    update(params: AssessmentsContextUpdateOptions, callback?: (error: Error | null, item?: AssessmentsInstance) => any): Promise<AssessmentsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface AssessmentsContextSolution {
    assessmentSid: string;
}
export declare class AssessmentsContextImpl implements AssessmentsContext {
    protected _version: V1;
    protected _solution: AssessmentsContextSolution;
    protected _uri: string;
    constructor(_version: V1, assessmentSid: string);
    update(params: AssessmentsContextUpdateOptions, callback?: (error: Error | null, item?: AssessmentsInstance) => any): Promise<AssessmentsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): AssessmentsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface AssessmentsPayload extends TwilioResponsePayload {
    assessments: AssessmentsResource[];
}
interface AssessmentsResource {
    account_sid: string;
    assessment_sid: string;
    offset: number;
    report: boolean;
    weight: number;
    agent_id: string;
    segment_id: string;
    user_name: string;
    user_email: string;
    answer_text: string;
    answer_id: string;
    assessment: any;
    timestamp: number;
    url: string;
}
export declare class AssessmentsInstance {
    protected _version: V1;
    protected _solution: AssessmentsContextSolution;
    protected _context?: AssessmentsContext;
    constructor(_version: V1, payload: AssessmentsResource, assessmentSid?: string);
    /**
     * The unique SID identifier of the Account.
     */
    accountSid: string;
    /**
     * The SID of the assessment
     */
    assessmentSid: string;
    /**
     * Offset of the conversation
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
     * The id of the Agent
     */
    agentId: string;
    /**
     * Segment Id of conversation
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
     * The answer text selected by user
     */
    answerText: string;
    /**
     * The id of the answer selected by user
     */
    answerId: string;
    /**
     * Assessment Details associated with an assessment
     */
    assessment: any;
    timestamp: number;
    url: string;
    private get _proxy();
    /**
     * Update a AssessmentsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssessmentsInstance
     */
    update(params: AssessmentsContextUpdateOptions, callback?: (error: Error | null, item?: AssessmentsInstance) => any): Promise<AssessmentsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        assessmentSid: string;
        offset: number;
        report: boolean;
        weight: number;
        agentId: string;
        segmentId: string;
        userName: string;
        userEmail: string;
        answerText: string;
        answerId: string;
        assessment: any;
        timestamp: number;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface AssessmentsSolution {
}
export interface AssessmentsListInstance {
    _version: V1;
    _solution: AssessmentsSolution;
    _uri: string;
    (assessmentSid: string): AssessmentsContext;
    get(assessmentSid: string): AssessmentsContext;
    /**
     * Create a AssessmentsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed AssessmentsInstance
     */
    create(params: AssessmentsListInstanceCreateOptions, callback?: (error: Error | null, item?: AssessmentsInstance) => any): Promise<AssessmentsInstance>;
    /**
     * Streams AssessmentsInstance records from the API.
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
     * @param { AssessmentsListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: AssessmentsInstance, done: (err?: Error) => void) => void): void;
    each(params: AssessmentsListInstanceEachOptions, callback?: (item: AssessmentsInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of AssessmentsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: AssessmentsPage) => any): Promise<AssessmentsPage>;
    /**
     * Lists AssessmentsInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AssessmentsListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: AssessmentsInstance[]) => any): Promise<AssessmentsInstance[]>;
    list(params: AssessmentsListInstanceOptions, callback?: (error: Error | null, items: AssessmentsInstance[]) => any): Promise<AssessmentsInstance[]>;
    /**
     * Retrieve a single page of AssessmentsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { AssessmentsListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: AssessmentsPage) => any): Promise<AssessmentsPage>;
    page(params: AssessmentsListInstancePageOptions, callback?: (error: Error | null, items: AssessmentsPage) => any): Promise<AssessmentsPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function AssessmentsListInstance(version: V1): AssessmentsListInstance;
export declare class AssessmentsPage extends Page<V1, AssessmentsPayload, AssessmentsResource, AssessmentsInstance> {
    /**
     * Initialize the AssessmentsPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: AssessmentsSolution);
    /**
     * Build an instance of AssessmentsInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: AssessmentsResource): AssessmentsInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
