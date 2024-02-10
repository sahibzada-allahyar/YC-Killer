/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
/**
 * Options to pass to create a DeploymentInstance
 */
export interface DeploymentListInstanceCreateOptions {
    /** The SID of the Build for the Deployment. */
    buildSid?: string;
}
/**
 * Options to pass to each
 */
export interface DeploymentListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: DeploymentInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface DeploymentListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface DeploymentListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface DeploymentContext {
    /**
     * Fetch a DeploymentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeploymentInstance
     */
    fetch(callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface DeploymentContextSolution {
    serviceSid: string;
    environmentSid: string;
    sid: string;
}
export declare class DeploymentContextImpl implements DeploymentContext {
    protected _version: V1;
    protected _solution: DeploymentContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, environmentSid: string, sid: string);
    fetch(callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): DeploymentContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface DeploymentPayload extends TwilioResponsePayload {
    deployments: DeploymentResource[];
}
interface DeploymentResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    environment_sid: string;
    build_sid: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class DeploymentInstance {
    protected _version: V1;
    protected _solution: DeploymentContextSolution;
    protected _context?: DeploymentContext;
    constructor(_version: V1, payload: DeploymentResource, serviceSid: string, environmentSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Deployment resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Deployment resource.
     */
    accountSid: string;
    /**
     * The SID of the Service that the Deployment resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the Environment for the Deployment.
     */
    environmentSid: string;
    /**
     * The SID of the Build for the deployment.
     */
    buildSid: string;
    /**
     * The date and time in GMT when the Deployment resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the Deployment resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Deployment resource.
     */
    url: string;
    private get _proxy();
    /**
     * Fetch a DeploymentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeploymentInstance
     */
    fetch(callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        environmentSid: string;
        buildSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface DeploymentSolution {
    serviceSid: string;
    environmentSid: string;
}
export interface DeploymentListInstance {
    _version: V1;
    _solution: DeploymentSolution;
    _uri: string;
    (sid: string): DeploymentContext;
    get(sid: string): DeploymentContext;
    /**
     * Create a DeploymentInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeploymentInstance
     */
    create(callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Create a DeploymentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed DeploymentInstance
     */
    create(params: DeploymentListInstanceCreateOptions, callback?: (error: Error | null, item?: DeploymentInstance) => any): Promise<DeploymentInstance>;
    /**
     * Streams DeploymentInstance records from the API.
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
     * @param { DeploymentListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: DeploymentInstance, done: (err?: Error) => void) => void): void;
    each(params: DeploymentListInstanceEachOptions, callback?: (item: DeploymentInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of DeploymentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: DeploymentPage) => any): Promise<DeploymentPage>;
    /**
     * Lists DeploymentInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DeploymentListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: DeploymentInstance[]) => any): Promise<DeploymentInstance[]>;
    list(params: DeploymentListInstanceOptions, callback?: (error: Error | null, items: DeploymentInstance[]) => any): Promise<DeploymentInstance[]>;
    /**
     * Retrieve a single page of DeploymentInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { DeploymentListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: DeploymentPage) => any): Promise<DeploymentPage>;
    page(params: DeploymentListInstancePageOptions, callback?: (error: Error | null, items: DeploymentPage) => any): Promise<DeploymentPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function DeploymentListInstance(version: V1, serviceSid: string, environmentSid: string): DeploymentListInstance;
export declare class DeploymentPage extends Page<V1, DeploymentPayload, DeploymentResource, DeploymentInstance> {
    /**
     * Initialize the DeploymentPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: DeploymentSolution);
    /**
     * Build an instance of DeploymentInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: DeploymentResource): DeploymentInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
