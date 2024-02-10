/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../../base/Page";
import Response from "../../../../../http/response";
import V2010 from "../../../V2010";
import { CredentialListInstance as CredentialListInstanceImport } from "./credentialList/credential";
/**
 * Options to pass to update a CredentialListInstance
 */
export interface CredentialListContextUpdateOptions {
    /** A human readable descriptive text for a CredentialList, up to 64 characters long. */
    friendlyName: string;
}
/**
 * Options to pass to create a CredentialListInstance
 */
export interface CredentialListListInstanceCreateOptions {
    /** A human readable descriptive text that describes the CredentialList, up to 64 characters long. */
    friendlyName: string;
}
/**
 * Options to pass to each
 */
export interface CredentialListListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: CredentialListInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface CredentialListListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface CredentialListListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface CredentialListContext {
    credentials: CredentialListInstanceImport;
    /**
     * Remove a CredentialListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CredentialListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CredentialListInstance
     */
    fetch(callback?: (error: Error | null, item?: CredentialListInstance) => any): Promise<CredentialListInstance>;
    /**
     * Update a CredentialListInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CredentialListInstance
     */
    update(params: CredentialListContextUpdateOptions, callback?: (error: Error | null, item?: CredentialListInstance) => any): Promise<CredentialListInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface CredentialListContextSolution {
    accountSid: string;
    sid: string;
}
export declare class CredentialListContextImpl implements CredentialListContext {
    protected _version: V2010;
    protected _solution: CredentialListContextSolution;
    protected _uri: string;
    protected _credentials?: CredentialListInstanceImport;
    constructor(_version: V2010, accountSid: string, sid: string);
    get credentials(): CredentialListInstanceImport;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: CredentialListInstance) => any): Promise<CredentialListInstance>;
    update(params: CredentialListContextUpdateOptions, callback?: (error: Error | null, item?: CredentialListInstance) => any): Promise<CredentialListInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): CredentialListContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface CredentialListPayload extends TwilioResponsePayload {
    credential_lists: CredentialListResource[];
}
interface CredentialListResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    friendly_name: string;
    sid: string;
    subresource_uris: Record<string, string>;
    uri: string;
}
export declare class CredentialListInstance {
    protected _version: V2010;
    protected _solution: CredentialListContextSolution;
    protected _context?: CredentialListContext;
    constructor(_version: V2010, payload: CredentialListResource, accountSid: string, sid?: string);
    /**
     * The unique id of the [Account](https://www.twilio.com/docs/iam/api/account) that owns this resource.
     */
    accountSid: string;
    /**
     * The date that this resource was created, given as GMT in [RFC 2822](https://www.php.net/manual/en/class.datetime.php#datetime.constants.rfc2822) format.
     */
    dateCreated: Date;
    /**
     * The date that this resource was last updated, given as GMT in [RFC 2822](https://www.php.net/manual/en/class.datetime.php#datetime.constants.rfc2822) format.
     */
    dateUpdated: Date;
    /**
     * A human readable descriptive text that describes the CredentialList, up to 64 characters long.
     */
    friendlyName: string;
    /**
     * A 34 character string that uniquely identifies this resource.
     */
    sid: string;
    /**
     * A list of credentials associated with this credential list.
     */
    subresourceUris: Record<string, string>;
    /**
     * The URI for this resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Remove a CredentialListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a CredentialListInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CredentialListInstance
     */
    fetch(callback?: (error: Error | null, item?: CredentialListInstance) => any): Promise<CredentialListInstance>;
    /**
     * Update a CredentialListInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CredentialListInstance
     */
    update(params: CredentialListContextUpdateOptions, callback?: (error: Error | null, item?: CredentialListInstance) => any): Promise<CredentialListInstance>;
    /**
     * Access the credentials.
     */
    credentials(): CredentialListInstanceImport;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        friendlyName: string;
        sid: string;
        subresourceUris: Record<string, string>;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CredentialListSolution {
    accountSid: string;
}
export interface CredentialListListInstance {
    _version: V2010;
    _solution: CredentialListSolution;
    _uri: string;
    (sid: string): CredentialListContext;
    get(sid: string): CredentialListContext;
    /**
     * Create a CredentialListInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CredentialListInstance
     */
    create(params: CredentialListListInstanceCreateOptions, callback?: (error: Error | null, item?: CredentialListInstance) => any): Promise<CredentialListInstance>;
    /**
     * Streams CredentialListInstance records from the API.
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
     * @param { CredentialListListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: CredentialListInstance, done: (err?: Error) => void) => void): void;
    each(params: CredentialListListInstanceEachOptions, callback?: (item: CredentialListInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of CredentialListInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: CredentialListPage) => any): Promise<CredentialListPage>;
    /**
     * Lists CredentialListInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CredentialListListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: CredentialListInstance[]) => any): Promise<CredentialListInstance[]>;
    list(params: CredentialListListInstanceOptions, callback?: (error: Error | null, items: CredentialListInstance[]) => any): Promise<CredentialListInstance[]>;
    /**
     * Retrieve a single page of CredentialListInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { CredentialListListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: CredentialListPage) => any): Promise<CredentialListPage>;
    page(params: CredentialListListInstancePageOptions, callback?: (error: Error | null, items: CredentialListPage) => any): Promise<CredentialListPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CredentialListListInstance(version: V2010, accountSid: string): CredentialListListInstance;
export declare class CredentialListPage extends Page<V2010, CredentialListPayload, CredentialListResource, CredentialListInstance> {
    /**
     * Initialize the CredentialListPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2010, response: Response<string>, solution: CredentialListSolution);
    /**
     * Build an instance of CredentialListInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: CredentialListResource): CredentialListInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
