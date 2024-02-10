/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
/**
 * Options to pass to update a ConnectionPolicyTargetInstance
 */
export interface ConnectionPolicyTargetContextUpdateOptions {
    /** A descriptive string that you create to describe the resource. It is not unique and can be up to 255 characters long. */
    friendlyName?: string;
    /** The SIP address you want Twilio to route your calls to. This must be a `sip:` schema. `sips` is NOT supported. */
    target?: string;
    /** The relative importance of the target. Can be an integer from 0 to 65535, inclusive. The lowest number represents the most important target. */
    priority?: number;
    /** The value that determines the relative share of the load the Target should receive compared to other Targets with the same priority. Can be an integer from 1 to 65535, inclusive. Targets with higher values receive more load than those with lower ones with the same priority. */
    weight?: number;
    /** Whether the Target is enabled. */
    enabled?: boolean;
}
/**
 * Options to pass to create a ConnectionPolicyTargetInstance
 */
export interface ConnectionPolicyTargetListInstanceCreateOptions {
    /** The SIP address you want Twilio to route your calls to. This must be a `sip:` schema. `sips` is NOT supported. */
    target: string;
    /** A descriptive string that you create to describe the resource. It is not unique and can be up to 255 characters long. */
    friendlyName?: string;
    /** The relative importance of the target. Can be an integer from 0 to 65535, inclusive, and the default is 10. The lowest number represents the most important target. */
    priority?: number;
    /** The value that determines the relative share of the load the Target should receive compared to other Targets with the same priority. Can be an integer from 1 to 65535, inclusive, and the default is 10. Targets with higher values receive more load than those with lower ones with the same priority. */
    weight?: number;
    /** Whether the Target is enabled. The default is `true`. */
    enabled?: boolean;
}
/**
 * Options to pass to each
 */
export interface ConnectionPolicyTargetListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: ConnectionPolicyTargetInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface ConnectionPolicyTargetListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface ConnectionPolicyTargetListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface ConnectionPolicyTargetContext {
    /**
     * Remove a ConnectionPolicyTargetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ConnectionPolicyTargetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyTargetInstance
     */
    fetch(callback?: (error: Error | null, item?: ConnectionPolicyTargetInstance) => any): Promise<ConnectionPolicyTargetInstance>;
    /**
     * Update a ConnectionPolicyTargetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyTargetInstance
     */
    update(callback?: (error: Error | null, item?: ConnectionPolicyTargetInstance) => any): Promise<ConnectionPolicyTargetInstance>;
    /**
     * Update a ConnectionPolicyTargetInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyTargetInstance
     */
    update(params: ConnectionPolicyTargetContextUpdateOptions, callback?: (error: Error | null, item?: ConnectionPolicyTargetInstance) => any): Promise<ConnectionPolicyTargetInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ConnectionPolicyTargetContextSolution {
    connectionPolicySid: string;
    sid: string;
}
export declare class ConnectionPolicyTargetContextImpl implements ConnectionPolicyTargetContext {
    protected _version: V1;
    protected _solution: ConnectionPolicyTargetContextSolution;
    protected _uri: string;
    constructor(_version: V1, connectionPolicySid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: ConnectionPolicyTargetInstance) => any): Promise<ConnectionPolicyTargetInstance>;
    update(params?: ConnectionPolicyTargetContextUpdateOptions | ((error: Error | null, item?: ConnectionPolicyTargetInstance) => any), callback?: (error: Error | null, item?: ConnectionPolicyTargetInstance) => any): Promise<ConnectionPolicyTargetInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ConnectionPolicyTargetContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ConnectionPolicyTargetPayload extends TwilioResponsePayload {
    targets: ConnectionPolicyTargetResource[];
}
interface ConnectionPolicyTargetResource {
    account_sid: string;
    connection_policy_sid: string;
    sid: string;
    friendly_name: string;
    target: string;
    priority: number;
    weight: number;
    enabled: boolean;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class ConnectionPolicyTargetInstance {
    protected _version: V1;
    protected _solution: ConnectionPolicyTargetContextSolution;
    protected _context?: ConnectionPolicyTargetContext;
    constructor(_version: V1, payload: ConnectionPolicyTargetResource, connectionPolicySid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Target resource.
     */
    accountSid: string;
    /**
     * The SID of the Connection Policy that owns the Target.
     */
    connectionPolicySid: string;
    /**
     * The unique string that we created to identify the Target resource.
     */
    sid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The SIP address you want Twilio to route your calls to. This must be a `sip:` schema. `sips` is NOT supported.
     */
    target: string;
    /**
     * The relative importance of the target. Can be an integer from 0 to 65535, inclusive, and the default is 10. The lowest number represents the most important target.
     */
    priority: number;
    /**
     * The value that determines the relative share of the load the Target should receive compared to other Targets with the same priority. Can be an integer from 1 to 65535, inclusive, and the default is 10. Targets with higher values receive more load than those with lower ones with the same priority.
     */
    weight: number;
    /**
     * Whether the target is enabled. The default is `true`.
     */
    enabled: boolean;
    /**
     * The date and time in GMT when the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a ConnectionPolicyTargetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a ConnectionPolicyTargetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyTargetInstance
     */
    fetch(callback?: (error: Error | null, item?: ConnectionPolicyTargetInstance) => any): Promise<ConnectionPolicyTargetInstance>;
    /**
     * Update a ConnectionPolicyTargetInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyTargetInstance
     */
    update(callback?: (error: Error | null, item?: ConnectionPolicyTargetInstance) => any): Promise<ConnectionPolicyTargetInstance>;
    /**
     * Update a ConnectionPolicyTargetInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyTargetInstance
     */
    update(params: ConnectionPolicyTargetContextUpdateOptions, callback?: (error: Error | null, item?: ConnectionPolicyTargetInstance) => any): Promise<ConnectionPolicyTargetInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        connectionPolicySid: string;
        sid: string;
        friendlyName: string;
        target: string;
        priority: number;
        weight: number;
        enabled: boolean;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ConnectionPolicyTargetSolution {
    connectionPolicySid: string;
}
export interface ConnectionPolicyTargetListInstance {
    _version: V1;
    _solution: ConnectionPolicyTargetSolution;
    _uri: string;
    (sid: string): ConnectionPolicyTargetContext;
    get(sid: string): ConnectionPolicyTargetContext;
    /**
     * Create a ConnectionPolicyTargetInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ConnectionPolicyTargetInstance
     */
    create(params: ConnectionPolicyTargetListInstanceCreateOptions, callback?: (error: Error | null, item?: ConnectionPolicyTargetInstance) => any): Promise<ConnectionPolicyTargetInstance>;
    /**
     * Streams ConnectionPolicyTargetInstance records from the API.
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
     * @param { ConnectionPolicyTargetListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: ConnectionPolicyTargetInstance, done: (err?: Error) => void) => void): void;
    each(params: ConnectionPolicyTargetListInstanceEachOptions, callback?: (item: ConnectionPolicyTargetInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of ConnectionPolicyTargetInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: ConnectionPolicyTargetPage) => any): Promise<ConnectionPolicyTargetPage>;
    /**
     * Lists ConnectionPolicyTargetInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ConnectionPolicyTargetListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: ConnectionPolicyTargetInstance[]) => any): Promise<ConnectionPolicyTargetInstance[]>;
    list(params: ConnectionPolicyTargetListInstanceOptions, callback?: (error: Error | null, items: ConnectionPolicyTargetInstance[]) => any): Promise<ConnectionPolicyTargetInstance[]>;
    /**
     * Retrieve a single page of ConnectionPolicyTargetInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { ConnectionPolicyTargetListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: ConnectionPolicyTargetPage) => any): Promise<ConnectionPolicyTargetPage>;
    page(params: ConnectionPolicyTargetListInstancePageOptions, callback?: (error: Error | null, items: ConnectionPolicyTargetPage) => any): Promise<ConnectionPolicyTargetPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ConnectionPolicyTargetListInstance(version: V1, connectionPolicySid: string): ConnectionPolicyTargetListInstance;
export declare class ConnectionPolicyTargetPage extends Page<V1, ConnectionPolicyTargetPayload, ConnectionPolicyTargetResource, ConnectionPolicyTargetInstance> {
    /**
     * Initialize the ConnectionPolicyTargetPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: ConnectionPolicyTargetSolution);
    /**
     * Build an instance of ConnectionPolicyTargetInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: ConnectionPolicyTargetResource): ConnectionPolicyTargetInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
