/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
/**
 * Options to pass to create a ChannelInstance
 */
export interface ChannelListInstanceCreateOptions {
    /** The SID of the Flex Flow. */
    flexFlowSid: string;
    /** The `identity` value that uniquely identifies the new resource\\\'s chat User. */
    identity: string;
    /** The chat participant\\\'s friendly name. */
    chatUserFriendlyName: string;
    /** The chat channel\\\'s friendly name. */
    chatFriendlyName: string;
    /** The Target Contact Identity, for example the phone number of an SMS. */
    target?: string;
    /** The chat channel\\\'s unique name. */
    chatUniqueName?: string;
    /** The pre-engagement data. */
    preEngagementData?: string;
    /** The SID of the TaskRouter Task. Only valid when integration type is `task`. `null` for integration types `studio` & `external` */
    taskSid?: string;
    /** The Task attributes to be added for the TaskRouter Task. */
    taskAttributes?: string;
    /** Whether to create the channel as long-lived. */
    longLived?: boolean;
}
/**
 * Options to pass to each
 */
export interface ChannelListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ChannelInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ChannelListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ChannelListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ChannelContext {
    /**
     * Remove a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ChannelContextSolution {
    sid: string;
}
export declare class ChannelContextImpl implements ChannelContext {
    protected _version: V1;
    protected _solution: ChannelContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ChannelContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ChannelPayload extends TwilioResponsePayload {
    flex_chat_channels: ChannelResource[];
}
interface ChannelResource {
    account_sid: string;
    flex_flow_sid: string;
    sid: string;
    user_sid: string;
    task_sid: string;
    url: string;
    date_created: Date;
    date_updated: Date;
}
export declare class ChannelInstance {
    protected _version: V1;
    protected _solution: ChannelContextSolution;
    protected _context?: ChannelContext;
    constructor(_version: V1, payload: ChannelResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Channel resource and owns this Workflow.
     */
    accountSid: string;
    /**
     * The SID of the Flex Flow.
     */
    flexFlowSid: string;
    /**
     * The unique string that we created to identify the Channel resource.
     */
    sid: string;
    /**
     * The SID of the chat user.
     */
    userSid: string;
    /**
     * The SID of the TaskRouter Task. Only valid when integration type is `task`. `null` for integration types `studio` & `external`
     */
    taskSid: string;
    /**
     * The absolute URL of the Flex chat channel resource.
     */
    url: string;
    /**
     * The date and time in GMT when the Flex chat channel was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the Flex chat channel was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    private get _proxy();
    /**
     * Remove a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ChannelInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    fetch(callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        flexFlowSid: string;
        sid: string;
        userSid: string;
        taskSid: string;
        url: string;
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ChannelSolution {
}
export interface ChannelListInstance {
    _version: V1;
    _solution: ChannelSolution;
    _uri: string;
    (sid: string): ChannelContext;
    get(sid: string): ChannelContext;
    /**
     * Create a ChannelInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ChannelInstance
     */
    create(params: ChannelListInstanceCreateOptions, callback?: (error: Error | null, item?: ChannelInstance) => any): Promise<ChannelInstance>;
    /**
     * Streams ChannelInstance records from the API.
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
     * @param { ChannelListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ChannelInstance, done: (err?: Error) => void) => void): void;
    each(params: ChannelListInstanceEachOptions, callback?: (item: ChannelInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ChannelPage) => any): Promise<ChannelPage>;
    /**
     * Lists ChannelInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ChannelListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ChannelInstance[]) => any): Promise<ChannelInstance[]>;
    list(params: ChannelListInstanceOptions, callback?: (error: Error | null, items: ChannelInstance[]) => any): Promise<ChannelInstance[]>;
    /**
     * Retrieve a single page of ChannelInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ChannelListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ChannelPage) => any): Promise<ChannelPage>;
    page(params: ChannelListInstancePageOptions, callback?: (error: Error | null, items: ChannelPage) => any): Promise<ChannelPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ChannelListInstance(version: V1): ChannelListInstance;
export declare class ChannelPage extends Page<V1, ChannelPayload, ChannelResource, ChannelInstance> {
    /**
     * Initialize the ChannelPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ChannelSolution);
    /**
     * Build an instance of ChannelInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ChannelResource): ChannelInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
