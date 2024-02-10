/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V1 from "../../../V1";
/**
 * Options to pass to update a VariableInstance
 */
export interface VariableContextUpdateOptions {
    /** A string by which the Variable resource can be referenced. It can be a maximum of 128 characters. */
    key?: string;
    /** A string that contains the actual value of the Variable. It can be a maximum of 450 bytes in size. */
    value?: string;
}
/**
 * Options to pass to create a VariableInstance
 */
export interface VariableListInstanceCreateOptions {
    /** A string by which the Variable resource can be referenced. It can be a maximum of 128 characters. */
    key: string;
    /** A string that contains the actual value of the Variable. It can be a maximum of 450 bytes in size. */
    value: string;
}
/**
 * Options to pass to each
 */
export interface VariableListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: VariableInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface VariableListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface VariableListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface VariableContext {
    /**
     * Remove a VariableInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a VariableInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VariableInstance
     */
    fetch(callback?: (error: Error | null, item?: VariableInstance) => any): Promise<VariableInstance>;
    /**
     * Update a VariableInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VariableInstance
     */
    update(callback?: (error: Error | null, item?: VariableInstance) => any): Promise<VariableInstance>;
    /**
     * Update a VariableInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VariableInstance
     */
    update(params: VariableContextUpdateOptions, callback?: (error: Error | null, item?: VariableInstance) => any): Promise<VariableInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface VariableContextSolution {
    serviceSid: string;
    environmentSid: string;
    sid: string;
}
export declare class VariableContextImpl implements VariableContext {
    protected _version: V1;
    protected _solution: VariableContextSolution;
    protected _uri: string;
    constructor(_version: V1, serviceSid: string, environmentSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: VariableInstance) => any): Promise<VariableInstance>;
    update(params?: VariableContextUpdateOptions | ((error: Error | null, item?: VariableInstance) => any), callback?: (error: Error | null, item?: VariableInstance) => any): Promise<VariableInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): VariableContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface VariablePayload extends TwilioResponsePayload {
    variables: VariableResource[];
}
interface VariableResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    environment_sid: string;
    key: string;
    value: string;
    date_created: Date;
    date_updated: Date;
    url: string;
}
export declare class VariableInstance {
    protected _version: V1;
    protected _solution: VariableContextSolution;
    protected _context?: VariableContext;
    constructor(_version: V1, payload: VariableResource, serviceSid: string, environmentSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Variable resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Variable resource.
     */
    accountSid: string;
    /**
     * The SID of the Service that the Variable resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the Environment in which the Variable exists.
     */
    environmentSid: string;
    /**
     * A string by which the Variable resource can be referenced.
     */
    key: string;
    /**
     * A string that contains the actual value of the Variable.
     */
    value: string;
    /**
     * The date and time in GMT when the Variable resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the Variable resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Variable resource.
     */
    url: string;
    private get _proxy();
    /**
     * Remove a VariableInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a VariableInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VariableInstance
     */
    fetch(callback?: (error: Error | null, item?: VariableInstance) => any): Promise<VariableInstance>;
    /**
     * Update a VariableInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VariableInstance
     */
    update(callback?: (error: Error | null, item?: VariableInstance) => any): Promise<VariableInstance>;
    /**
     * Update a VariableInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VariableInstance
     */
    update(params: VariableContextUpdateOptions, callback?: (error: Error | null, item?: VariableInstance) => any): Promise<VariableInstance>;
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
        key: string;
        value: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface VariableSolution {
    serviceSid: string;
    environmentSid: string;
}
export interface VariableListInstance {
    _version: V1;
    _solution: VariableSolution;
    _uri: string;
    (sid: string): VariableContext;
    get(sid: string): VariableContext;
    /**
     * Create a VariableInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VariableInstance
     */
    create(params: VariableListInstanceCreateOptions, callback?: (error: Error | null, item?: VariableInstance) => any): Promise<VariableInstance>;
    /**
     * Streams VariableInstance records from the API.
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
     * @param { VariableListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: VariableInstance, done: (err?: Error) => void) => void): void;
    each(params: VariableListInstanceEachOptions, callback?: (item: VariableInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of VariableInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: VariablePage) => any): Promise<VariablePage>;
    /**
     * Lists VariableInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { VariableListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: VariableInstance[]) => any): Promise<VariableInstance[]>;
    list(params: VariableListInstanceOptions, callback?: (error: Error | null, items: VariableInstance[]) => any): Promise<VariableInstance[]>;
    /**
     * Retrieve a single page of VariableInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { VariableListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: VariablePage) => any): Promise<VariablePage>;
    page(params: VariableListInstancePageOptions, callback?: (error: Error | null, items: VariablePage) => any): Promise<VariablePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function VariableListInstance(version: V1, serviceSid: string, environmentSid: string): VariableListInstance;
export declare class VariablePage extends Page<V1, VariablePayload, VariableResource, VariableInstance> {
    /**
     * Initialize the VariablePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: VariableSolution);
    /**
     * Build an instance of VariableInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: VariableResource): VariableInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
