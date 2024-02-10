/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to update a ActivityInstance
 */
export interface ActivityContextUpdateOptions {
    /** A descriptive string that you create to describe the Activity resource. It can be up to 64 characters long. These names are used to calculate and expose statistics about Workers, and provide visibility into the state of each Worker. Examples of friendly names include: `on-call`, `break`, and `email`. */
    friendlyName?: string;
}
/**
 * Options to pass to create a ActivityInstance
 */
export interface ActivityListInstanceCreateOptions {
    /** A descriptive string that you create to describe the Activity resource. It can be up to 64 characters long. These names are used to calculate and expose statistics about Workers, and provide visibility into the state of each Worker. Examples of friendly names include: `on-call`, `break`, and `email`. */
    friendlyName: string;
    /** Whether the Worker should be eligible to receive a Task when it occupies the Activity. A value of `true`, `1`, or `yes` specifies the Activity is available. All other values specify that it is not. The value cannot be changed after the Activity is created. */
    available?: boolean;
}
/**
 * Options to pass to each
 */
export interface ActivityListInstanceEachOptions {
    /** The `friendly_name` of the Activity resources to read. */
    friendlyName?: string;
    /** Whether return only Activity resources that are available or unavailable. A value of `true` returns only available activities. Values of \'1\' or `yes` also indicate `true`. All other values represent `false` and return activities that are unavailable. */
    available?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ActivityInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ActivityListInstanceOptions {
    /** The `friendly_name` of the Activity resources to read. */
    friendlyName?: string;
    /** Whether return only Activity resources that are available or unavailable. A value of `true` returns only available activities. Values of \'1\' or `yes` also indicate `true`. All other values represent `false` and return activities that are unavailable. */
    available?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ActivityListInstancePageOptions {
    /** The `friendly_name` of the Activity resources to read. */
    friendlyName?: string;
    /** Whether return only Activity resources that are available or unavailable. A value of `true` returns only available activities. Values of \'1\' or `yes` also indicate `true`. All other values represent `false` and return activities that are unavailable. */
    available?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ActivityContext {
    /**
     * Remove a ActivityInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ActivityInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ActivityInstance
     */
    fetch(callback?: (error: Error | null, item?: ActivityInstance) => any): Promise<ActivityInstance>;
    /**
     * Update a ActivityInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ActivityInstance
     */
    update(callback?: (error: Error | null, item?: ActivityInstance) => any): Promise<ActivityInstance>;
    /**
     * Update a ActivityInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ActivityInstance
     */
    update(params: ActivityContextUpdateOptions, callback?: (error: Error | null, item?: ActivityInstance) => any): Promise<ActivityInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ActivityContextSolution {
    workspaceSid: string;
    sid: string;
}
export declare class ActivityContextImpl implements ActivityContext {
    protected _version: V1;
    protected _solution: ActivityContextSolution;
    protected _uri: string;
    constructor(_version: V1, workspaceSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ActivityInstance) => any): Promise<ActivityInstance>;
    update(params?: ActivityContextUpdateOptions | ((error: Error | null, item?: ActivityInstance) => any), callback?: (error: Error | null, item?: ActivityInstance) => any): Promise<ActivityInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ActivityContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ActivityPayload extends TwilioResponsePayload {
    activities: ActivityResource[];
}
interface ActivityResource {
    account_sid: string;
    available: boolean;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    sid: string;
    workspace_sid: string;
    url: string;
    links: Record<string, string>;
}
export declare class ActivityInstance {
    protected _version: V1;
    protected _solution: ActivityContextSolution;
    protected _context?: ActivityContext;
    constructor(_version: V1, payload: ActivityResource, workspaceSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Activity resource.
     */
    accountSid: string;
    /**
     * Whether the Worker is eligible to receive a Task when it occupies the Activity. A value of `true`, `1`, or `yes` indicates the Activity is available. All other values indicate that it is not. The value cannot be changed after the Activity is created.
     */
    available: boolean;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The string that you assigned to describe the Activity resource.
     */
    friendlyName: string;
    /**
     * The unique string that we created to identify the Activity resource.
     */
    sid: string;
    /**
     * The SID of the Workspace that contains the Activity.
     */
    workspaceSid: string;
    /**
     * The absolute URL of the Activity resource.
     */
    url: string;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a ActivityInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ActivityInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ActivityInstance
     */
    fetch(callback?: (error: Error | null, item?: ActivityInstance) => any): Promise<ActivityInstance>;
    /**
     * Update a ActivityInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ActivityInstance
     */
    update(callback?: (error: Error | null, item?: ActivityInstance) => any): Promise<ActivityInstance>;
    /**
     * Update a ActivityInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ActivityInstance
     */
    update(params: ActivityContextUpdateOptions, callback?: (error: Error | null, item?: ActivityInstance) => any): Promise<ActivityInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        available: boolean;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        sid: string;
        workspaceSid: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ActivitySolution {
    workspaceSid: string;
}
export interface ActivityListInstance {
    _version: V1;
    _solution: ActivitySolution;
    _uri: string;
    (sid: string): ActivityContext;
    get(sid: string): ActivityContext;
    /**
     * Create a ActivityInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ActivityInstance
     */
    create(params: ActivityListInstanceCreateOptions, callback?: (error: Error | null, item?: ActivityInstance) => any): Promise<ActivityInstance>;
    /**
     * Streams ActivityInstance records from the API.
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
     * @param { ActivityListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ActivityInstance, done: (err?: Error) => void) => void): void;
    each(params: ActivityListInstanceEachOptions, callback?: (item: ActivityInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ActivityInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ActivityPage) => any): Promise<ActivityPage>;
    /**
     * Lists ActivityInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ActivityListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ActivityInstance[]) => any): Promise<ActivityInstance[]>;
    list(params: ActivityListInstanceOptions, callback?: (error: Error | null, items: ActivityInstance[]) => any): Promise<ActivityInstance[]>;
    /**
     * Retrieve a single page of ActivityInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ActivityListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ActivityPage) => any): Promise<ActivityPage>;
    page(params: ActivityListInstancePageOptions, callback?: (error: Error | null, items: ActivityPage) => any): Promise<ActivityPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ActivityListInstance(version: V1, workspaceSid: string): ActivityListInstance;
export declare class ActivityPage extends Page<V1, ActivityPayload, ActivityResource, ActivityInstance> {
    /**
     * Initialize the ActivityPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ActivitySolution);
    /**
     * Build an instance of ActivityInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ActivityResource): ActivityInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
