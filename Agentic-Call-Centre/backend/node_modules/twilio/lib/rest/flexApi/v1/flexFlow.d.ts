/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
export type FlexFlowChannelType = "web" | "sms" | "facebook" | "whatsapp" | "line" | "custom";
export type FlexFlowIntegrationType = "studio" | "external" | "task";
/**
 * Options to pass to update a FlexFlowInstance
 */
export interface FlexFlowContextUpdateOptions {
    /** A descriptive string that you create to describe the Flex Flow resource. */
    friendlyName?: string;
    /** The SID of the chat service. */
    chatServiceSid?: string;
    /**  */
    channelType?: FlexFlowChannelType;
    /** The channel contact\\\'s Identity. */
    contactIdentity?: string;
    /** Whether the new Flex Flow is enabled. */
    enabled?: boolean;
    /**  */
    integrationType?: FlexFlowIntegrationType;
    /** The SID of the Studio Flow. Required when `integrationType` is `studio`. */
    "integration.flowSid"?: string;
    /** The URL of the external webhook. Required when `integrationType` is `external`. */
    "integration.url"?: string;
    /** The Workspace SID for a new Task. Required when `integrationType` is `task`. */
    "integration.workspaceSid"?: string;
    /** The Workflow SID for a new Task. Required when `integrationType` is `task`. */
    "integration.workflowSid"?: string;
    /** The Task Channel SID (TCXXXX) or unique name (e.g., `sms`) to use for the Task that will be created. Applicable and required when `integrationType` is `task`. The default value is `default`. */
    "integration.channel"?: string;
    /** The Task timeout in seconds for a new Task. Default is 86,400 seconds (24 hours). Optional when `integrationType` is `task`, not applicable otherwise. */
    "integration.timeout"?: number;
    /** The Task priority of a new Task. The default priority is 0. Optional when `integrationType` is `task`, not applicable otherwise. */
    "integration.priority"?: number;
    /** In the context of outbound messaging, defines whether to create a Task immediately (and therefore reserve the conversation to current agent), or delay Task creation until the customer sends the first response. Set to false to create immediately, true to delay Task creation. This setting is only applicable for outbound messaging. */
    "integration.creationOnMessage"?: boolean;
    /** When enabled, Flex will keep the chat channel active so that it may be used for subsequent interactions with a contact identity. Defaults to `false`. */
    longLived?: boolean;
    /** When enabled, the Messaging Channel Janitor will remove active Proxy sessions if the associated Task is deleted outside of the Flex UI. Defaults to `false`. */
    janitorEnabled?: boolean;
    /** The number of times to retry the Studio Flow or webhook in case of failure. Takes integer values from 0 to 3 with the default being 3. Optional when `integrationType` is `studio` or `external`, not applicable otherwise. */
    "integration.retryCount"?: number;
}
/**
 * Options to pass to create a FlexFlowInstance
 */
export interface FlexFlowListInstanceCreateOptions {
    /** A descriptive string that you create to describe the Flex Flow resource. */
    friendlyName: string;
    /** The SID of the chat service. */
    chatServiceSid: string;
    /**  */
    channelType: FlexFlowChannelType;
    /** The channel contact\\\'s Identity. */
    contactIdentity?: string;
    /** Whether the new Flex Flow is enabled. */
    enabled?: boolean;
    /**  */
    integrationType?: FlexFlowIntegrationType;
    /** The SID of the Studio Flow. Required when `integrationType` is `studio`. */
    "integration.flowSid"?: string;
    /** The URL of the external webhook. Required when `integrationType` is `external`. */
    "integration.url"?: string;
    /** The Workspace SID for a new Task. Required when `integrationType` is `task`. */
    "integration.workspaceSid"?: string;
    /** The Workflow SID for a new Task. Required when `integrationType` is `task`. */
    "integration.workflowSid"?: string;
    /** The Task Channel SID (TCXXXX) or unique name (e.g., `sms`) to use for the Task that will be created. Applicable and required when `integrationType` is `task`. The default value is `default`. */
    "integration.channel"?: string;
    /** The Task timeout in seconds for a new Task. Default is 86,400 seconds (24 hours). Optional when `integrationType` is `task`, not applicable otherwise. */
    "integration.timeout"?: number;
    /** The Task priority of a new Task. The default priority is 0. Optional when `integrationType` is `task`, not applicable otherwise. */
    "integration.priority"?: number;
    /** In the context of outbound messaging, defines whether to create a Task immediately (and therefore reserve the conversation to current agent), or delay Task creation until the customer sends the first response. Set to false to create immediately, true to delay Task creation. This setting is only applicable for outbound messaging. */
    "integration.creationOnMessage"?: boolean;
    /** When enabled, Flex will keep the chat channel active so that it may be used for subsequent interactions with a contact identity. Defaults to `false`. */
    longLived?: boolean;
    /** When enabled, the Messaging Channel Janitor will remove active Proxy sessions if the associated Task is deleted outside of the Flex UI. Defaults to `false`. */
    janitorEnabled?: boolean;
    /** The number of times to retry the Studio Flow or webhook in case of failure. Takes integer values from 0 to 3 with the default being 3. Optional when `integrationType` is `studio` or `external`, not applicable otherwise. */
    "integration.retryCount"?: number;
}
/**
 * Options to pass to each
 */
export interface FlexFlowListInstanceEachOptions {
    /** The `friendly_name` of the Flex Flow resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: FlexFlowInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface FlexFlowListInstanceOptions {
    /** The `friendly_name` of the Flex Flow resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface FlexFlowListInstancePageOptions {
    /** The `friendly_name` of the Flex Flow resources to read. */
    friendlyName?: string;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface FlexFlowContext {
    /**
     * Remove a FlexFlowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a FlexFlowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlexFlowInstance
     */
    fetch(callback?: (error: Error | null, item?: FlexFlowInstance) => any): Promise<FlexFlowInstance>;
    /**
     * Update a FlexFlowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlexFlowInstance
     */
    update(callback?: (error: Error | null, item?: FlexFlowInstance) => any): Promise<FlexFlowInstance>;
    /**
     * Update a FlexFlowInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlexFlowInstance
     */
    update(params: FlexFlowContextUpdateOptions, callback?: (error: Error | null, item?: FlexFlowInstance) => any): Promise<FlexFlowInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface FlexFlowContextSolution {
    sid: string;
}
export declare class FlexFlowContextImpl implements FlexFlowContext {
    protected _version: V1;
    protected _solution: FlexFlowContextSolution;
    protected _uri: string;
    constructor(_version: V1, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: FlexFlowInstance) => any): Promise<FlexFlowInstance>;
    update(params?: FlexFlowContextUpdateOptions | ((error: Error | null, item?: FlexFlowInstance) => any), callback?: (error: Error | null, item?: FlexFlowInstance) => any): Promise<FlexFlowInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): FlexFlowContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface FlexFlowPayload extends TwilioResponsePayload {
    flex_flows: FlexFlowResource[];
}
interface FlexFlowResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    sid: string;
    friendly_name: string;
    chat_service_sid: string;
    channel_type: FlexFlowChannelType;
    contact_identity: string;
    enabled: boolean;
    integration_type: FlexFlowIntegrationType;
    integration: any;
    long_lived: boolean;
    janitor_enabled: boolean;
    url: string;
}
export declare class FlexFlowInstance {
    protected _version: V1;
    protected _solution: FlexFlowContextSolution;
    protected _context?: FlexFlowContext;
    constructor(_version: V1, payload: FlexFlowResource, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Flex Flow resource and owns this Workflow.
     */
    accountSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The unique string that we created to identify the Flex Flow resource.
     */
    sid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The SID of the chat service.
     */
    chatServiceSid: string;
    channelType: FlexFlowChannelType;
    /**
     * The channel contact\'s Identity.
     */
    contactIdentity: string;
    /**
     * Whether the Flex Flow is enabled.
     */
    enabled: boolean;
    integrationType: FlexFlowIntegrationType;
    /**
     * An object that contains specific parameters for the integration.
     */
    integration: any;
    /**
     * When enabled, Flex will keep the chat channel active so that it may be used for subsequent interactions with a contact identity. Defaults to `false`.
     */
    longLived: boolean;
    /**
     * When enabled, the Messaging Channel Janitor will remove active Proxy sessions if the associated Task is deleted outside of the Flex UI. Defaults to `false`.
     */
    janitorEnabled: boolean;
    /**
     * The absolute URL of the Flex Flow resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a FlexFlowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a FlexFlowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlexFlowInstance
     */
    fetch(callback?: (error: Error | null, item?: FlexFlowInstance) => any): Promise<FlexFlowInstance>;
    /**
     * Update a FlexFlowInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlexFlowInstance
     */
    update(callback?: (error: Error | null, item?: FlexFlowInstance) => any): Promise<FlexFlowInstance>;
    /**
     * Update a FlexFlowInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlexFlowInstance
     */
    update(params: FlexFlowContextUpdateOptions, callback?: (error: Error | null, item?: FlexFlowInstance) => any): Promise<FlexFlowInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        sid: string;
        friendlyName: string;
        chatServiceSid: string;
        channelType: FlexFlowChannelType;
        contactIdentity: string;
        enabled: boolean;
        integrationType: FlexFlowIntegrationType;
        integration: any;
        longLived: boolean;
        janitorEnabled: boolean;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface FlexFlowSolution {
}
export interface FlexFlowListInstance {
    _version: V1;
    _solution: FlexFlowSolution;
    _uri: string;
    (sid: string): FlexFlowContext;
    get(sid: string): FlexFlowContext;
    /**
     * Create a FlexFlowInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FlexFlowInstance
     */
    create(params: FlexFlowListInstanceCreateOptions, callback?: (error: Error | null, item?: FlexFlowInstance) => any): Promise<FlexFlowInstance>;
    /**
     * Streams FlexFlowInstance records from the API.
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
     * @param { FlexFlowListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: FlexFlowInstance, done: (err?: Error) => void) => void): void;
    each(params: FlexFlowListInstanceEachOptions, callback?: (item: FlexFlowInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of FlexFlowInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: FlexFlowPage) => any): Promise<FlexFlowPage>;
    /**
     * Lists FlexFlowInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FlexFlowListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: FlexFlowInstance[]) => any): Promise<FlexFlowInstance[]>;
    list(params: FlexFlowListInstanceOptions, callback?: (error: Error | null, items: FlexFlowInstance[]) => any): Promise<FlexFlowInstance[]>;
    /**
     * Retrieve a single page of FlexFlowInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { FlexFlowListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: FlexFlowPage) => any): Promise<FlexFlowPage>;
    page(params: FlexFlowListInstancePageOptions, callback?: (error: Error | null, items: FlexFlowPage) => any): Promise<FlexFlowPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function FlexFlowListInstance(version: V1): FlexFlowListInstance;
export declare class FlexFlowPage extends Page<V1, FlexFlowPayload, FlexFlowResource, FlexFlowInstance> {
    /**
     * Initialize the FlexFlowPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: FlexFlowSolution);
    /**
     * Build an instance of FlexFlowInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: FlexFlowResource): FlexFlowInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
