/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V1 from "../../V1";
import { BuildStatusListInstance } from "./build/buildStatus";
export type BuildRuntime = "node8" | "node10" | "node12" | "node14" | "node16" | "node18";
export type BuildStatus = "building" | "completed" | "failed";
/**
 * Options to pass to create a BuildInstance
 */
export interface BuildListInstanceCreateOptions {
    /** The list of Asset Version resource SIDs to include in the Build. */
    assetVersions?: Array<string>;
    /** The list of the Function Version resource SIDs to include in the Build. */
    functionVersions?: Array<string>;
    /** A list of objects that describe the Dependencies included in the Build. Each object contains the `name` and `version` of the dependency. */
    dependencies?: string;
    /** The Runtime version that will be used to run the Build resource when it is deployed. */
    runtime?: string;
}
/**
 * Options to pass to each
 */
export interface BuildListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: BuildInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface BuildListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface BuildListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface BuildContext {
    buildStatus: BuildStatusListInstance;
    /**
     * Remove a BuildInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a BuildInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BuildInstance
     */
    fetch(callback?: (error: Error | null, item?: BuildInstance) => any): Promise<BuildInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface BuildContextSolution {
    serviceSid: string;
    sid: string;
}
export declare class BuildContextImpl implements BuildContext {
    protected _version: V1;
    protected _solution: BuildContextSolution;
    protected _uri: string;
    protected _buildStatus?: BuildStatusListInstance;
    constructor(_version: V1, serviceSid: string, sid: string);
    get buildStatus(): BuildStatusListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: BuildInstance) => any): Promise<BuildInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): BuildContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface BuildPayload extends TwilioResponsePayload {
    builds: BuildResource[];
}
interface BuildResource {
    sid: string;
    account_sid: string;
    service_sid: string;
    status: BuildStatus;
    asset_versions: Array<any>;
    function_versions: Array<any>;
    dependencies: Array<any>;
    runtime: BuildRuntime;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class BuildInstance {
    protected _version: V1;
    protected _solution: BuildContextSolution;
    protected _context?: BuildContext;
    constructor(_version: V1, payload: BuildResource, serviceSid: string, sid?: string);
    /**
     * The unique string that we created to identify the Build resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Build resource.
     */
    accountSid: string;
    /**
     * The SID of the Service that the Build resource is associated with.
     */
    serviceSid: string;
    status: BuildStatus;
    /**
     * The list of Asset Version resource SIDs that are included in the Build.
     */
    assetVersions: Array<any>;
    /**
     * The list of Function Version resource SIDs that are included in the Build.
     */
    functionVersions: Array<any>;
    /**
     * A list of objects that describe the Dependencies included in the Build. Each object contains the `name` and `version` of the dependency.
     */
    dependencies: Array<any>;
    runtime: BuildRuntime;
    /**
     * The date and time in GMT when the Build resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the Build resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Build resource.
     */
    url: string;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a BuildInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a BuildInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BuildInstance
     */
    fetch(callback?: (error: Error | null, item?: BuildInstance) => any): Promise<BuildInstance>;
    /**
     * Access the buildStatus.
     */
    buildStatus(): BuildStatusListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        serviceSid: string;
        status: BuildStatus;
        assetVersions: any[];
        functionVersions: any[];
        dependencies: any[];
        runtime: BuildRuntime;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface BuildSolution {
    serviceSid: string;
}
export interface BuildListInstance {
    _version: V1;
    _solution: BuildSolution;
    _uri: string;
    (sid: string): BuildContext;
    get(sid: string): BuildContext;
    /**
     * Create a BuildInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BuildInstance
     */
    create(callback?: (error: Error | null, item?: BuildInstance) => any): Promise<BuildInstance>;
    /**
     * Create a BuildInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BuildInstance
     */
    create(params: BuildListInstanceCreateOptions, callback?: (error: Error | null, item?: BuildInstance) => any): Promise<BuildInstance>;
    /**
     * Streams BuildInstance records from the API.
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
     * @param { BuildListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: BuildInstance, done: (err?: Error) => void) => void): void;
    each(params: BuildListInstanceEachOptions, callback?: (item: BuildInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of BuildInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: BuildPage) => any): Promise<BuildPage>;
    /**
     * Lists BuildInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BuildListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: BuildInstance[]) => any): Promise<BuildInstance[]>;
    list(params: BuildListInstanceOptions, callback?: (error: Error | null, items: BuildInstance[]) => any): Promise<BuildInstance[]>;
    /**
     * Retrieve a single page of BuildInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BuildListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: BuildPage) => any): Promise<BuildPage>;
    page(params: BuildListInstancePageOptions, callback?: (error: Error | null, items: BuildPage) => any): Promise<BuildPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BuildListInstance(version: V1, serviceSid: string): BuildListInstance;
export declare class BuildPage extends Page<V1, BuildPayload, BuildResource, BuildInstance> {
    /**
     * Initialize the BuildPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: BuildSolution);
    /**
     * Build an instance of BuildInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: BuildResource): BuildInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
