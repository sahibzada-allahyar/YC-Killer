/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to remove a InsightsQuestionnairesInstance
 */
export interface InsightsQuestionnairesContextRemoveOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
}
/**
 * Options to pass to fetch a InsightsQuestionnairesInstance
 */
export interface InsightsQuestionnairesContextFetchOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
}
/**
 * Options to pass to update a InsightsQuestionnairesInstance
 */
export interface InsightsQuestionnairesContextUpdateOptions {
    /** The flag to enable or disable questionnaire */
    active: boolean;
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The name of this questionnaire */
    name?: string;
    /** The description of this questionnaire */
    description?: string;
    /** The list of questions sids under a questionnaire */
    questionSids?: Array<string>;
}
/**
 * Options to pass to create a InsightsQuestionnairesInstance
 */
export interface InsightsQuestionnairesListInstanceCreateOptions {
    /** The name of this questionnaire */
    name: string;
    /** The Authorization HTTP request header */
    authorization?: string;
    /** The description of this questionnaire */
    description?: string;
    /** The flag to enable or disable questionnaire */
    active?: boolean;
    /** The list of questions sids under a questionnaire */
    questionSids?: Array<string>;
}
/**
 * Options to pass to each
 */
export interface InsightsQuestionnairesListInstanceEachOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** Flag indicating whether to include inactive questionnaires or not */
    includeInactive?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: InsightsQuestionnairesInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface InsightsQuestionnairesListInstanceOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** Flag indicating whether to include inactive questionnaires or not */
    includeInactive?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface InsightsQuestionnairesListInstancePageOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** Flag indicating whether to include inactive questionnaires or not */
    includeInactive?: boolean;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface InsightsQuestionnairesContext {
    /**
     * Remove a InsightsQuestionnairesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a InsightsQuestionnairesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesInstance
     */
    remove(params: InsightsQuestionnairesContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a InsightsQuestionnairesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesInstance
     */
    fetch(callback?: (error: Error | null, item?: InsightsQuestionnairesInstance) => any): Promise<InsightsQuestionnairesInstance>;
    /**
     * Fetch a InsightsQuestionnairesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesInstance
     */
    fetch(params: InsightsQuestionnairesContextFetchOptions, callback?: (error: Error | null, item?: InsightsQuestionnairesInstance) => any): Promise<InsightsQuestionnairesInstance>;
    /**
     * Update a InsightsQuestionnairesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesInstance
     */
    update(params: InsightsQuestionnairesContextUpdateOptions, callback?: (error: Error | null, item?: InsightsQuestionnairesInstance) => any): Promise<InsightsQuestionnairesInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface InsightsQuestionnairesContextSolution {
    questionnaireSid: string;
}
export declare class InsightsQuestionnairesContextImpl implements InsightsQuestionnairesContext {
    protected _version: V1;
    protected _solution: InsightsQuestionnairesContextSolution;
    protected _uri: string;
    constructor(_version: V1, questionnaireSid: string);
    remove(params?: InsightsQuestionnairesContextRemoveOptions | ((error: Error | null, item?: boolean) => any), callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(params?: InsightsQuestionnairesContextFetchOptions | ((error: Error | null, item?: InsightsQuestionnairesInstance) => any), callback?: (error: Error | null, item?: InsightsQuestionnairesInstance) => any): Promise<InsightsQuestionnairesInstance>;
    update(params: InsightsQuestionnairesContextUpdateOptions, callback?: (error: Error | null, item?: InsightsQuestionnairesInstance) => any): Promise<InsightsQuestionnairesInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): InsightsQuestionnairesContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface InsightsQuestionnairesPayload extends TwilioResponsePayload {
    questionnaires: InsightsQuestionnairesResource[];
}
interface InsightsQuestionnairesResource {
    account_sid: string;
    questionnaire_sid: string;
    name: string;
    description: string;
    active: boolean;
    questions: Array<any>;
    url: string;
}
export declare class InsightsQuestionnairesInstance {
    protected _version: V1;
    protected _solution: InsightsQuestionnairesContextSolution;
    protected _context?: InsightsQuestionnairesContext;
    constructor(_version: V1, payload: InsightsQuestionnairesResource, questionnaireSid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Flex Insights resource and owns this resource.
     */
    accountSid: string;
    /**
     * The sid of this questionnaire
     */
    questionnaireSid: string;
    /**
     * The name of this category.
     */
    name: string;
    /**
     * The description of this questionnaire
     */
    description: string;
    /**
     * The flag to enable or disable questionnaire
     */
    active: boolean;
    /**
     * The list of questions with category for a questionnaire
     */
    questions: Array<any>;
    url: string;
    private get _proxy();
    /**
     * Remove a InsightsQuestionnairesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Remove a InsightsQuestionnairesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesInstance
     */
    remove(params: InsightsQuestionnairesContextRemoveOptions, callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a InsightsQuestionnairesInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesInstance
     */
    fetch(callback?: (error: Error | null, item?: InsightsQuestionnairesInstance) => any): Promise<InsightsQuestionnairesInstance>;
    /**
     * Fetch a InsightsQuestionnairesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesInstance
     */
    fetch(params: InsightsQuestionnairesContextFetchOptions, callback?: (error: Error | null, item?: InsightsQuestionnairesInstance) => any): Promise<InsightsQuestionnairesInstance>;
    /**
     * Update a InsightsQuestionnairesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesInstance
     */
    update(params: InsightsQuestionnairesContextUpdateOptions, callback?: (error: Error | null, item?: InsightsQuestionnairesInstance) => any): Promise<InsightsQuestionnairesInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        questionnaireSid: string;
        name: string;
        description: string;
        active: boolean;
        questions: any[];
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface InsightsQuestionnairesSolution {
}
export interface InsightsQuestionnairesListInstance {
    _version: V1;
    _solution: InsightsQuestionnairesSolution;
    _uri: string;
    (questionnaireSid: string): InsightsQuestionnairesContext;
    get(questionnaireSid: string): InsightsQuestionnairesContext;
    /**
     * Create a InsightsQuestionnairesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed InsightsQuestionnairesInstance
     */
    create(params: InsightsQuestionnairesListInstanceCreateOptions, callback?: (error: Error | null, item?: InsightsQuestionnairesInstance) => any): Promise<InsightsQuestionnairesInstance>;
    /**
     * Streams InsightsQuestionnairesInstance records from the API.
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
     * @param { InsightsQuestionnairesListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: InsightsQuestionnairesInstance, done: (err?: Error) => void) => void): void;
    each(params: InsightsQuestionnairesListInstanceEachOptions, callback?: (item: InsightsQuestionnairesInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of InsightsQuestionnairesInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: InsightsQuestionnairesPage) => any): Promise<InsightsQuestionnairesPage>;
    /**
     * Lists InsightsQuestionnairesInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InsightsQuestionnairesListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: InsightsQuestionnairesInstance[]) => any): Promise<InsightsQuestionnairesInstance[]>;
    list(params: InsightsQuestionnairesListInstanceOptions, callback?: (error: Error | null, items: InsightsQuestionnairesInstance[]) => any): Promise<InsightsQuestionnairesInstance[]>;
    /**
     * Retrieve a single page of InsightsQuestionnairesInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InsightsQuestionnairesListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: InsightsQuestionnairesPage) => any): Promise<InsightsQuestionnairesPage>;
    page(params: InsightsQuestionnairesListInstancePageOptions, callback?: (error: Error | null, items: InsightsQuestionnairesPage) => any): Promise<InsightsQuestionnairesPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InsightsQuestionnairesListInstance(version: V1): InsightsQuestionnairesListInstance;
export declare class InsightsQuestionnairesPage extends Page<V1, InsightsQuestionnairesPayload, InsightsQuestionnairesResource, InsightsQuestionnairesInstance> {
    /**
     * Initialize the InsightsQuestionnairesPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: InsightsQuestionnairesSolution);
    /**
     * Build an instance of InsightsQuestionnairesInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: InsightsQuestionnairesResource): InsightsQuestionnairesInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
