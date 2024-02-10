/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { EngagementContextListInstance } from "./engagement/engagementContext";
import { StepListInstance } from "./engagement/step";
export type EngagementStatus = "active" | "ended";
/**
 * Options to pass to create a EngagementInstance
 */
export interface EngagementListInstanceCreateOptions {
    /** The Contact phone number to start a Studio Flow Engagement, available as variable `{{contact.channel.address}}`. */
    to: string;
    /** The Twilio phone number to send messages or initiate calls from during the Flow Engagement. Available as variable `{{flow.channel.address}}` */
    from: string;
    /** A JSON string we will add to your flow\\\'s context and that you can access as variables inside your flow. For example, if you pass in `Parameters={\\\'name\\\':\\\'Zeke\\\'}` then inside a widget you can reference the variable `{{flow.data.name}}` which will return the string \\\'Zeke\\\'. Note: the JSON value must explicitly be passed as a string, not as a hash object. Depending on your particular HTTP library, you may need to add quotes or URL encode your JSON string. */
    parameters?: any;
}
/**
 * Options to pass to each
 */
export interface EngagementListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: EngagementInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface EngagementListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface EngagementListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface EngagementContext {
    engagementContext: EngagementContextListInstance;
    steps: StepListInstance;
    /**
     * Remove a EngagementInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a EngagementInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EngagementInstance
     */
    fetch(callback?: (error: Error | null, item?: EngagementInstance) => any): Promise<EngagementInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface EngagementContextSolution {
    flowSid: string;
    sid: string;
}
export declare class EngagementContextImpl implements EngagementContext {
    protected _version: V1;
    protected _solution: EngagementContextSolution;
    protected _uri: string;
    protected _engagementContext?: EngagementContextListInstance;
    protected _steps?: StepListInstance;
    constructor(_version: V1, flowSid: string, sid: string);
    get engagementContext(): EngagementContextListInstance;
    get steps(): StepListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: EngagementInstance) => any): Promise<EngagementInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): EngagementContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface EngagementPayload extends TwilioResponsePayload {
    engagements: EngagementResource[];
}
interface EngagementResource {
    sid: string;
    account_sid: string;
    flow_sid: string;
    contact_sid: string;
    contact_channel_address: string;
    context: any;
    status: EngagementStatus;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class EngagementInstance {
    protected _version: V1;
    protected _solution: EngagementContextSolution;
    protected _context?: EngagementContext;
    constructor(_version: V1, payload: EngagementResource, flowSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Engagement resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Engagement resource.
     */
    accountSid: string;
    /**
     * The SID of the Flow.
     */
    flowSid: string;
    /**
     * The SID of the Contact.
     */
    contactSid: string;
    /**
     * The phone number, SIP address or Client identifier that triggered this Engagement. Phone numbers are in E.164 format (+16175551212). SIP addresses are formatted as `name@company.com`. Client identifiers are formatted `client:name`.
     */
    contactChannelAddress: string;
    /**
     * The current state of the execution flow. As your flow executes, we save the state in a flow context. Your widgets can access the data in the flow context as variables, either in configuration fields or in text areas as variable substitution.
     */
    context: any;
    status: EngagementStatus;
    /**
     * The date and time in GMT when the Engagement was created in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the Engagement was updated in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    /**
     * The URLs of the Engagement\'s nested resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a EngagementInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a EngagementInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EngagementInstance
     */
    fetch(callback?: (error: Error | null, item?: EngagementInstance) => any): Promise<EngagementInstance>;
    /**
     * Access the engagementContext.
     */
    engagementContext(): EngagementContextListInstance;
    /**
     * Access the steps.
     */
    steps(): StepListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        flowSid: string;
        contactSid: string;
        contactChannelAddress: string;
        context: any;
        status: EngagementStatus;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface EngagementSolution {
    flowSid: string;
}
export interface EngagementListInstance {
    _version: V1;
    _solution: EngagementSolution;
    _uri: string;
    (sid: string): EngagementContext;
    get(sid: string): EngagementContext;
    /**
     * Create a EngagementInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EngagementInstance
     */
    create(params: EngagementListInstanceCreateOptions, callback?: (error: Error | null, item?: EngagementInstance) => any): Promise<EngagementInstance>;
    /**
     * Streams EngagementInstance records from the API.
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
     * @param { EngagementListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: EngagementInstance, done: (err?: Error) => void) => void): void;
    each(params: EngagementListInstanceEachOptions, callback?: (item: EngagementInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of EngagementInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: EngagementPage) => any): Promise<EngagementPage>;
    /**
     * Lists EngagementInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EngagementListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: EngagementInstance[]) => any): Promise<EngagementInstance[]>;
    list(params: EngagementListInstanceOptions, callback?: (error: Error | null, items: EngagementInstance[]) => any): Promise<EngagementInstance[]>;
    /**
     * Retrieve a single page of EngagementInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EngagementListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: EngagementPage) => any): Promise<EngagementPage>;
    page(params: EngagementListInstancePageOptions, callback?: (error: Error | null, items: EngagementPage) => any): Promise<EngagementPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function EngagementListInstance(version: V1, flowSid: string): EngagementListInstance;
export declare class EngagementPage extends Page<V1, EngagementPayload, EngagementResource, EngagementInstance> {
    /**
     * Initialize the EngagementPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: EngagementSolution);
    /**
     * Build an instance of EngagementInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: EngagementResource): EngagementInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
