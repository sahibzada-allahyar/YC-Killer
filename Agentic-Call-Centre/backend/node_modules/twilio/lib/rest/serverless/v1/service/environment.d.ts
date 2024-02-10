/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { DeploymentListInstance } from "./environment/deployment";
import { LogListInstance } from "./environment/log";
import { VariableListInstance } from "./environment/variable";
/**
 * Options to pass to create a EnvironmentInstance
 */
export interface EnvironmentListInstanceCreateOptions {
    /** A user-defined string that uniquely identifies the Environment resource. It can be a maximum of 100 characters. */
    uniqueName: string;
    /** A URL-friendly name that represents the environment and forms part of the domain name. It can be a maximum of 16 characters. */
    domainSuffix?: string;
}
/**
 * Options to pass to each
 */
export interface EnvironmentListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: EnvironmentInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface EnvironmentListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface EnvironmentListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface EnvironmentContext {
    deployments: DeploymentListInstance;
    logs: LogListInstance;
    variables: VariableListInstance;
    /**
     * Remove a EnvironmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a EnvironmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EnvironmentInstance
     */
    fetch(callback?: (error: Error | null, item?: EnvironmentInstance) => any): Promise<EnvironmentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface EnvironmentContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class EnvironmentContextImpl implements EnvironmentContext {
    protected _version: V1;
    protected _solution: EnvironmentContextSolution;
    protected _uri: string;
    protected _deployments?: DeploymentListInstance;
    protected _logs?: LogListInstance;
    protected _variables?: VariableListInstance;
    constructor(_version: V1, serviceSid: string, sid: string);
    get deployments(): DeploymentListInstance;
    get logs(): LogListInstance;
    get variables(): VariableListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: EnvironmentInstance) => any): Promise<EnvironmentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): EnvironmentContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface EnvironmentPayload extends TwilioResponsePayload {
    environments: EnvironmentResource[];
}
interface EnvironmentResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    build_sid: string;
    unique_name: string;
    domain_suffix: string;
    domain_name: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class EnvironmentInstance {
    protected _version: V1;
    protected _solution: EnvironmentContextSolution;
    protected _context?: EnvironmentContext;
    constructor(_version: V1, payload: EnvironmentResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Environment resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Environment resource.
     */
    accountSid: string;
    /**
     * The SID of the Service that the Environment resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the build deployed in the environment.
     */
    buildSid: string;
    /**
     * A user-defined string that uniquely identifies the Environment resource.
     */
    uniqueName: string;
    /**
     * A URL-friendly name that represents the environment and forms part of the domain name.
     */
    domainSuffix: string;
    /**
     * The domain name for all Functions and Assets deployed in the Environment, using the Service unique name, a randomly-generated Service suffix, and an optional Environment domain suffix.
     */
    domainName: string;
    /**
     * The date and time in GMT when the Environment resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the Environment resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Environment resource.
     */
    url: string;
    /**
     * The URLs of the Environment resource\'s nested resources.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a EnvironmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a EnvironmentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EnvironmentInstance
     */
    fetch(callback?: (error: Error | null, item?: EnvironmentInstance) => any): Promise<EnvironmentInstance>;
    /**
     * Access the deployments.
     */
    deployments(): DeploymentListInstance;
    /**
     * Access the logs.
     */
    logs(): LogListInstance;
    /**
     * Access the variables.
     */
    variables(): VariableListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        buildSid: string;
        uniqueName: string;
        domainSuffix: string;
        domainName: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface EnvironmentSolution {
    serviceSid: string;
}
export interface EnvironmentListInstance {
    _version: V1;
    _solution: EnvironmentSolution;
    _uri: string;
    (sid: string): EnvironmentContext;
    get(sid: string): EnvironmentContext;
    /**
     * Create a EnvironmentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed EnvironmentInstance
     */
    create(params: EnvironmentListInstanceCreateOptions, callback?: (error: Error | null, item?: EnvironmentInstance) => any): Promise<EnvironmentInstance>;
    /**
     * Streams EnvironmentInstance records from the API.
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
     * @param { EnvironmentListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: EnvironmentInstance, done: (err?: Error) => void) => void): void;
    each(params: EnvironmentListInstanceEachOptions, callback?: (item: EnvironmentInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of EnvironmentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: EnvironmentPage) => any): Promise<EnvironmentPage>;
    /**
     * Lists EnvironmentInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EnvironmentListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: EnvironmentInstance[]) => any): Promise<EnvironmentInstance[]>;
    list(params: EnvironmentListInstanceOptions, callback?: (error: Error | null, items: EnvironmentInstance[]) => any): Promise<EnvironmentInstance[]>;
    /**
     * Retrieve a single page of EnvironmentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { EnvironmentListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: EnvironmentPage) => any): Promise<EnvironmentPage>;
    page(params: EnvironmentListInstancePageOptions, callback?: (error: Error | null, items: EnvironmentPage) => any): Promise<EnvironmentPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function EnvironmentListInstance(version: V1, serviceSid: string): EnvironmentListInstance;
export declare class EnvironmentPage extends Page<V1, EnvironmentPayload, EnvironmentResource, EnvironmentInstance> {
    /**
     * Initialize the EnvironmentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: EnvironmentSolution);
    /**
     * Build an instance of EnvironmentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: EnvironmentResource): EnvironmentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
