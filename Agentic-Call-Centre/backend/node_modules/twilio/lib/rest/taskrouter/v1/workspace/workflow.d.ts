/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { WorkflowCumulativeStatisticsListInstance } from "./workflow/workflowCumulativeStatistics";
import { WorkflowRealTimeStatisticsListInstance } from "./workflow/workflowRealTimeStatistics";
import { WorkflowStatisticsListInstance } from "./workflow/workflowStatistics";
/**
 * Options to pass to update a WorkflowInstance
 */
export interface WorkflowContextUpdateOptions {
    /** A descriptive string that you create to describe the Workflow resource. For example, `Inbound Call Workflow` or `2014 Outbound Campaign`. */
    friendlyName?: string;
    /** The URL from your application that will process task assignment events. See [Handling Task Assignment Callback](https://www.twilio.com/docs/taskrouter/handle-assignment-callbacks) for more details. */
    assignmentCallbackUrl?: string;
    /** The URL that we should call when a call to the `assignment_callback_url` fails. */
    fallbackAssignmentCallbackUrl?: string;
    /** A JSON string that contains the rules to apply to the Workflow. See [Configuring Workflows](https://www.twilio.com/docs/taskrouter/workflow-configuration) for more information. */
    configuration?: string;
    /** How long TaskRouter will wait for a confirmation response from your application after it assigns a Task to a Worker. Can be up to `86,400` (24 hours) and the default is `120`. */
    taskReservationTimeout?: number;
    /** Whether or not to re-evaluate Tasks. The default is `false`, which means Tasks in the Workflow will not be processed through the assignment loop again. */
    reEvaluateTasks?: string;
}
/**
 * Options to pass to create a WorkflowInstance
 */
export interface WorkflowListInstanceCreateOptions {
    /** A descriptive string that you create to describe the Workflow resource. For example, `Inbound Call Workflow` or `2014 Outbound Campaign`. */
    friendlyName: string;
    /** A JSON string that contains the rules to apply to the Workflow. See [Configuring Workflows](https://www.twilio.com/docs/taskrouter/workflow-configuration) for more information. */
    configuration: string;
    /** The URL from your application that will process task assignment events. See [Handling Task Assignment Callback](https://www.twilio.com/docs/taskrouter/handle-assignment-callbacks) for more details. */
    assignmentCallbackUrl?: string;
    /** The URL that we should call when a call to the `assignment_callback_url` fails. */
    fallbackAssignmentCallbackUrl?: string;
    /** How long TaskRouter will wait for a confirmation response from your application after it assigns a Task to a Worker. Can be up to `86,400` (24 hours) and the default is `120`. */
    taskReservationTimeout?: number;
}
/**
 * Options to pass to each
 */
export interface WorkflowListInstanceEachOptions {
    /** The `friendly_name` of the Workflow resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: WorkflowInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface WorkflowListInstanceOptions {
    /** The `friendly_name` of the Workflow resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface WorkflowListInstancePageOptions {
    /** The `friendly_name` of the Workflow resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface WorkflowContext {
    cumulativeStatistics: WorkflowCumulativeStatisticsListInstance;
    realTimeStatistics: WorkflowRealTimeStatisticsListInstance;
    statistics: WorkflowStatisticsListInstance;
    /**
     * Remove a WorkflowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a WorkflowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkflowInstance) => any): Promise<WorkflowInstance>;
    /**
     * Update a WorkflowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowInstance
     */
    update(callback?: (error: Error | null, item?: WorkflowInstance) => any): Promise<WorkflowInstance>;
    /**
     * Update a WorkflowInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowInstance
     */
    update(params: WorkflowContextUpdateOptions, callback?: (error: Error | null, item?: WorkflowInstance) => any): Promise<WorkflowInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface WorkflowContextSolution {
    workspaceSid: string;
    sid: string;
}
export declare class WorkflowContextImpl implements WorkflowContext {
    protected _version: V1;
    protected _solution: WorkflowContextSolution;
    protected _uri: string;
    protected _cumulativeStatistics?: WorkflowCumulativeStatisticsListInstance;
    protected _realTimeStatistics?: WorkflowRealTimeStatisticsListInstance;
    protected _statistics?: WorkflowStatisticsListInstance;
    constructor(_version: V1, workspaceSid: string, sid: string);
    get cumulativeStatistics(): WorkflowCumulativeStatisticsListInstance;
    get realTimeStatistics(): WorkflowRealTimeStatisticsListInstance;
    get statistics(): WorkflowStatisticsListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: WorkflowInstance) => any): Promise<WorkflowInstance>;
    update(params?: WorkflowContextUpdateOptions | ((error: Error | null, item?: WorkflowInstance) => any), callback?: (error: Error | null, item?: WorkflowInstance) => any): Promise<WorkflowInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): WorkflowContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface WorkflowPayload extends TwilioResponsePayload {
    workflows: WorkflowResource[];
}
interface WorkflowResource {
    account_sid: string;
    assignment_callback_url: string;
    configuration: string;
    date_created: Date;
    date_updated: Date;
    document_content_type: string;
    fallback_assignment_callback_url: string;
    friendly_name: string;
    sid: string;
    task_reservation_timeout: number;
    workspace_sid: string;
    url: string;
    links: Record<string, string>;
}
export declare class WorkflowInstance {
    protected _version: V1;
    protected _solution: WorkflowContextSolution;
    protected _context?: WorkflowContext;
    constructor(_version: V1, payload: WorkflowResource, workspaceSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Workflow resource.
     */
    accountSid: string;
    /**
     * The URL that we call when a task managed by the Workflow is assigned to a Worker. See Assignment Callback URL for more information.
     */
    assignmentCallbackUrl: string;
    /**
     * A JSON string that contains the Workflow\'s configuration. See [Configuring Workflows](https://www.twilio.com/docs/taskrouter/workflow-configuration) for more information.
     */
    configuration: string;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The MIME type of the document.
     */
    documentContentType: string;
    /**
     * The URL that we call when a call to the `assignment_callback_url` fails.
     */
    fallbackAssignmentCallbackUrl: string;
    /**
     * The string that you assigned to describe the Workflow resource. For example, `Customer Support` or `2014 Election Campaign`.
     */
    friendlyName: string;
    /**
     * The unique string that we created to identify the Workflow resource.
     */
    sid: string;
    /**
     * How long TaskRouter will wait for a confirmation response from your application after it assigns a Task to a Worker. Can be up to `86,400` (24 hours) and the default is `120`.
     */
    taskReservationTimeout: number;
    /**
     * The SID of the Workspace that contains the Workflow.
     */
    workspaceSid: string;
    /**
     * The absolute URL of the Workflow resource.
     */
    url: string;
    /**
     * The URLs of related resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a WorkflowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a WorkflowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowInstance
     */
    fetch(callback?: (error: Error | null, item?: WorkflowInstance) => any): Promise<WorkflowInstance>;
    /**
     * Update a WorkflowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowInstance
     */
    update(callback?: (error: Error | null, item?: WorkflowInstance) => any): Promise<WorkflowInstance>;
    /**
     * Update a WorkflowInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowInstance
     */
    update(params: WorkflowContextUpdateOptions, callback?: (error: Error | null, item?: WorkflowInstance) => any): Promise<WorkflowInstance>;
    /**
     * Access the cumulativeStatistics.
     */
    cumulativeStatistics(): WorkflowCumulativeStatisticsListInstance;
    /**
     * Access the realTimeStatistics.
     */
    realTimeStatistics(): WorkflowRealTimeStatisticsListInstance;
    /**
     * Access the statistics.
     */
    statistics(): WorkflowStatisticsListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        assignmentCallbackUrl: string;
        configuration: string;
        dateCreated: Date;
        dateUpdated: Date;
        documentContentType: string;
        fallbackAssignmentCallbackUrl: string;
        friendlyName: string;
        sid: string;
        taskReservationTimeout: number;
        workspaceSid: string;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface WorkflowSolution {
    workspaceSid: string;
}
export interface WorkflowListInstance {
    _version: V1;
    _solution: WorkflowSolution;
    _uri: string;
    (sid: string): WorkflowContext;
    get(sid: string): WorkflowContext;
    /**
     * Create a WorkflowInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed WorkflowInstance
     */
    create(params: WorkflowListInstanceCreateOptions, callback?: (error: Error | null, item?: WorkflowInstance) => any): Promise<WorkflowInstance>;
    /**
     * Streams WorkflowInstance records from the API.
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
     * @param { WorkflowListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: WorkflowInstance, done: (err?: Error) => void) => void): void;
    each(params: WorkflowListInstanceEachOptions, callback?: (item: WorkflowInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of WorkflowInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: WorkflowPage) => any): Promise<WorkflowPage>;
    /**
     * Lists WorkflowInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { WorkflowListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: WorkflowInstance[]) => any): Promise<WorkflowInstance[]>;
    list(params: WorkflowListInstanceOptions, callback?: (error: Error | null, items: WorkflowInstance[]) => any): Promise<WorkflowInstance[]>;
    /**
     * Retrieve a single page of WorkflowInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { WorkflowListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: WorkflowPage) => any): Promise<WorkflowPage>;
    page(params: WorkflowListInstancePageOptions, callback?: (error: Error | null, items: WorkflowPage) => any): Promise<WorkflowPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function WorkflowListInstance(version: V1, workspaceSid: string): WorkflowListInstance;
export declare class WorkflowPage extends Page<V1, WorkflowPayload, WorkflowResource, WorkflowInstance> {
    /**
     * Initialize the WorkflowPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: WorkflowSolution);
    /**
     * Build an instance of WorkflowInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: WorkflowResource): WorkflowInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
