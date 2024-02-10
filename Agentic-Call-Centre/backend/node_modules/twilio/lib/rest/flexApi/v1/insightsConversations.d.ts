/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to each
 */
export interface InsightsConversationsListInstanceEachOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** Unique Id of the segment for which conversation details needs to be fetched */
    segmentId?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: InsightsConversationsInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface InsightsConversationsListInstanceOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** Unique Id of the segment for which conversation details needs to be fetched */
    segmentId?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface InsightsConversationsListInstancePageOptions {
    /** The Authorization HTTP request header */
    authorization?: string;
    /** Unique Id of the segment for which conversation details needs to be fetched */
    segmentId?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface InsightsConversationsSolution {
}
export interface InsightsConversationsListInstance {
    _version: V1;
    _solution: InsightsConversationsSolution;
    _uri: string;
    /**
     * Streams InsightsConversationsInstance records from the API.
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
     * @param { InsightsConversationsListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: InsightsConversationsInstance, done: (err?: Error) => void) => void): void;
    each(params: InsightsConversationsListInstanceEachOptions, callback?: (item: InsightsConversationsInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of InsightsConversationsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: InsightsConversationsPage) => any): Promise<InsightsConversationsPage>;
    /**
     * Lists InsightsConversationsInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InsightsConversationsListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: InsightsConversationsInstance[]) => any): Promise<InsightsConversationsInstance[]>;
    list(params: InsightsConversationsListInstanceOptions, callback?: (error: Error | null, items: InsightsConversationsInstance[]) => any): Promise<InsightsConversationsInstance[]>;
    /**
     * Retrieve a single page of InsightsConversationsInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { InsightsConversationsListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: InsightsConversationsPage) => any): Promise<InsightsConversationsPage>;
    page(params: InsightsConversationsListInstancePageOptions, callback?: (error: Error | null, items: InsightsConversationsPage) => any): Promise<InsightsConversationsPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function InsightsConversationsListInstance(version: V1): InsightsConversationsListInstance;
interface InsightsConversationsPayload extends TwilioResponsePayload {
    conversations: InsightsConversationsResource[];
}
interface InsightsConversationsResource {
    account_id: string;
    conversation_id: string;
    segment_count: number;
    segments: Array<any>;
}
export declare class InsightsConversationsInstance {
    protected _version: V1;
    constructor(_version: V1, payload: InsightsConversationsResource);
    /**
     * The id of the account.
     */
    accountId: string;
    /**
     * The unique id of the conversation
     */
    conversationId: string;
    /**
     * The count of segments for a conversation
     */
    segmentCount: number;
    /**
     * The Segments of a conversation
     */
    segments: Array<any>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountId: string;
        conversationId: string;
        segmentCount: number;
        segments: any[];
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export declare class InsightsConversationsPage extends Page<V1, InsightsConversationsPayload, InsightsConversationsResource, InsightsConversationsInstance> {
    /**
     * Initialize the InsightsConversationsPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: InsightsConversationsSolution);
    /**
     * Build an instance of InsightsConversationsInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: InsightsConversationsResource): InsightsConversationsInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
