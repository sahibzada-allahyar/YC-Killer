/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to remove a InsightsQuestionnairesQuestionInstance
 */
export interface InsightsQuestionnairesQuestionContextRemoveOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
}
/**
 * Options to pass to update a InsightsQuestionnairesQuestionInstance
 */
export interface InsightsQuestionnairesQuestionContextUpdateOptions {
    /** The flag to enable for disable NA for answer. */
    allowNa: boolean;
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The SID of the category */
    categorySid?: string;
    /** The question. */
    question?: string;
    /** The description for the question. */
    description?: string;
    /** The answer_set for the question. */
    answerSetId?: string;
}
/**
 * Options to pass to create a InsightsQuestionnairesQuestionInstance
 */
export interface InsightsQuestionnairesQuestionListInstanceCreateOptions {
    /** The SID of the category */
    categorySid: string;
    /** The question. */
    question: string;
    /** The answer_set for the question. */
    answerSetId: string;
    /** The flag to enable for disable NA for answer. */
    allowNa: boolean;
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The description for the question. */
    description?: string;
}
/**
 * Options to pass to each
 */
export interface InsightsQuestionnairesQuestionListInstanceEachOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The list of category SIDs */
    categorySid?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: InsightsQuestionnairesQuestionInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface InsightsQuestionnairesQuestionListInstanceOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The list of category SIDs */
    categorySid?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface InsightsQuestionnairesQuestionListInstancePageOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The list of category SIDs */
    categorySid?: Array<string>;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface InsightsQuestionnairesQuestionContext {
    /**
     * Remove a InsightsQuestionnairesQuestionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a InsightsQuestionnairesQuestionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesQuestionInstance
     */
    remove(params: InsightsQuestionnairesQuestionContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Update a InsightsQuestionnairesQuestionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesQuestionInstance
     */
    update(params: InsightsQuestionnairesQuestionContextUpdateOptions, callback?: (error: Error | null, item?: InsightsQuestionnairesQuestionInstance) => any): Promise<InsightsQuestionnairesQuestionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface InsightsQuestionnairesQuestionContextSolution {
    questionSid: string;
}
export declare class InsightsQuestionnairesQuestionContextImpl implements InsightsQuestionnairesQuestionContext {
    protected _version: V1;
    protected _solution: InsightsQuestionnairesQuestionContextSolution;
    protected _uri: string;
    constructor(_version: V1, questionSid: string);
    remove(params?: InsightsQuestionnairesQuestionContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    update(params: InsightsQuestionnairesQuestionContextUpdateOptions, callback?: (error: Error | null, item?: InsightsQuestionnairesQuestionInstance) => any): Promise<InsightsQuestionnairesQuestionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): InsightsQuestionnairesQuestionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface InsightsQuestionnairesQuestionPayload extends TwilioResponsePayload {
    questions: InsightsQuestionnairesQuestionResource[];
}
interface InsightsQuestionnairesQuestionResource {
    account_sid: string;
    question_sid: string;
    question: string;
    description: string;
    category: any;
    answer_set_id: string;
    allow_na: boolean;
    usage: number;
    answer_set: any;
    url: string;
}
export declare class InsightsQuestionnairesQuestionInstance {
    protected _version: V1;
    protected _solution: InsightsQuestionnairesQuestionContextSolution;
    protected _context?: InsightsQuestionnairesQuestionContext;
    constructor(_version: V1, payload: InsightsQuestionnairesQuestionResource, questionSid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Flex Insights resource and owns this resource.
     */
    accountSid: string;
    /**
     * The SID of the question
     */
    questionSid: string;
    /**
     * The question.
     */
    question: string;
    /**
     * The description for the question.
     */
    description: string;
    /**
     * The Category for the question.
     */
    category: any;
    /**
     * The answer_set for the question.
     */
    answerSetId: string;
    /**
     * The flag  to enable for disable NA for answer.
     */
    allowNa: boolean;
    /**
     * Integer value that tells a particular question is used by how many questionnaires
     */
    usage: number;
    /**
     * Set of answers for the question
     */
    answerSet: any;
    url: string;
    private get _proxy();
    /**
     * Remove a InsightsQuestionnairesQuestionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a InsightsQuestionnairesQuestionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesQuestionInstance
     */
    remove(params: InsightsQuestionnairesQuestionContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Update a InsightsQuestionnairesQuestionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesQuestionInstance
     */
    update(params: InsightsQuestionnairesQuestionContextUpdateOptions, callback?: (error: Error | null, item?: InsightsQuestionnairesQuestionInstance) => any): Promise<InsightsQuestionnairesQuestionInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        questionSid: string;
        question: string;
        description: string;
        category: any;
        answerSetId: string;
        allowNa: boolean;
        usage: number;
        answerSet: any;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface InsightsQuestionnairesQuestionSolution {
}
export interface InsightsQuestionnairesQuestionListInstance {
    _version: V1;
    _solution: InsightsQuestionnairesQuestionSolution;
    _uri: string;
    (questionSid: string): InsightsQuestionnairesQuestionContext;
    get(questionSid: string): InsightsQuestionnairesQuestionContext;
    /**
     * Create a InsightsQuestionnairesQuestionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesQuestionInstance
     */
    create(params: InsightsQuestionnairesQuestionListInstanceCreateOptions, callback?: (error: Error | null, item?: InsightsQuestionnairesQuestionInstance) => any): Promise<InsightsQuestionnairesQuestionInstance>;
    /**
     * Streams InsightsQuestionnairesQuestionInstance records from the API.
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
     * @param { InsightsQuestionnairesQuestionListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: InsightsQuestionnairesQuestionInstance, done: (err?: Error) => void) => void): void;
    each(params: InsightsQuestionnairesQuestionListInstanceEachOptions, callback?: (item: InsightsQuestionnairesQuestionInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of InsightsQuestionnairesQuestionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: InsightsQuestionnairesQuestionPage) => any): Promise<InsightsQuestionnairesQuestionPage>;
    /**
     * Lists InsightsQuestionnairesQuestionInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InsightsQuestionnairesQuestionListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: InsightsQuestionnairesQuestionInstance[]) => any): Promise<InsightsQuestionnairesQuestionInstance[]>;
    list(params: InsightsQuestionnairesQuestionListInstanceOptions, callback?: (error: Error | null, items: InsightsQuestionnairesQuestionInstance[]) => any): Promise<InsightsQuestionnairesQuestionInstance[]>;
    /**
     * Retrieve a single page of InsightsQuestionnairesQuestionInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InsightsQuestionnairesQuestionListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: InsightsQuestionnairesQuestionPage) => any): Promise<InsightsQuestionnairesQuestionPage>;
    page(params: InsightsQuestionnairesQuestionListInstancePageOptions, callback?: (error: Error | null, items: InsightsQuestionnairesQuestionPage) => any): Promise<InsightsQuestionnairesQuestionPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InsightsQuestionnairesQuestionListInstance(version: V1): InsightsQuestionnairesQuestionListInstance;
export declare class InsightsQuestionnairesQuestionPage extends Page<V1, InsightsQuestionnairesQuestionPayload, InsightsQuestionnairesQuestionResource, InsightsQuestionnairesQuestionInstance> {
    /**
     * Initialize the InsightsQuestionnairesQuestionPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: InsightsQuestionnairesQuestionSolution);
    /**
     * Build an instance of InsightsQuestionnairesQuestionInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: InsightsQuestionnairesQuestionResource): InsightsQuestionnairesQuestionInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
