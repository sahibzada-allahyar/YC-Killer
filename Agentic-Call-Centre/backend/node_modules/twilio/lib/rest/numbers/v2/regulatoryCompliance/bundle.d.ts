/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../../base/Page";
import Response from "../../../../http/response";
import V2 from "../../V2";
import { BundleCopyListInstance } from "./bundle/bundleCopy";
import { EvaluationListInstance } from "./bundle/evaluation";
import { ItemAssignmentListInstance } from "./bundle/itemAssignment";
import { ReplaceItemsListInstance } from "./bundle/replaceItems";
export type BundleEndUserType = "individual" | "business";
export type BundleSortBy = "valid-until" | "date-updated";
export type BundleSortDirection = "ASC" | "DESC";
export type BundleStatus = "draft" | "pending-review" | "in-review" | "twilio-rejected" | "twilio-approved" | "provisionally-approved";
/**
 * Options to pass to update a BundleInstance
 */
export interface BundleContextUpdateOptions {
    /**  */
    status?: BundleStatus;
    /** The URL we call to inform your application of status changes. */
    statusCallback?: string;
    /** The string that you assigned to describe the resource. */
    friendlyName?: string;
    /** The email address that will receive updates when the Bundle resource changes status. */
    email?: string;
}
/**
 * Options to pass to create a BundleInstance
 */
export interface BundleListInstanceCreateOptions {
    /** The string that you assigned to describe the resource. */
    friendlyName: string;
    /** The email address that will receive updates when the Bundle resource changes status. */
    email: string;
    /** The URL we call to inform your application of status changes. */
    statusCallback?: string;
    /** The unique string of a regulation that is associated to the Bundle resource. */
    regulationSid?: string;
    /** The [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the Bundle\\\'s phone number country ownership request. */
    isoCountry?: string;
    /**  */
    endUserType?: BundleEndUserType;
    /** The type of phone number of the Bundle\\\'s ownership request. Can be `local`, `mobile`, `national`, or `toll free`. */
    numberType?: string;
}
/**
 * Options to pass to each
 */
export interface BundleListInstanceEachOptions {
    /** The verification status of the Bundle resource. Please refer to [Bundle Statuses](https://www.twilio.com/docs/phone-numbers/regulatory/api/bundles#bundle-statuses) for more details. */
    status?: BundleStatus;
    /** The string that you assigned to describe the resource. The column can contain 255 variable characters. */
    friendlyName?: string;
    /** The unique string of a [Regulation resource](https://www.twilio.com/docs/phone-numbers/regulatory/api/regulations) that is associated to the Bundle resource. */
    regulationSid?: string;
    /** The 2-digit [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the Bundle\'s phone number country ownership request. */
    isoCountry?: string;
    /** The type of phone number of the Bundle\'s ownership request. Can be `local`, `mobile`, `national`, or `tollfree`. */
    numberType?: string;
    /** Indicates that the Bundle is a valid Bundle until a specified expiration date. */
    hasValidUntilDate?: boolean;
    /** Can be `valid-until` or `date-updated`. Defaults to `date-created`. */
    sortBy?: BundleSortBy;
    /** Default is `DESC`. Can be `ASC` or `DESC`. */
    sortDirection?: BundleSortDirection;
    /** Date to filter Bundles having their `valid_until_date` before or after the specified date. Can be `ValidUntilDate>=` or `ValidUntilDate<=`. Both can be used in conjunction as well. [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) is the acceptable date format. */
    validUntilDate?: Date;
    /** Date to filter Bundles having their `valid_until_date` before or after the specified date. Can be `ValidUntilDate>=` or `ValidUntilDate<=`. Both can be used in conjunction as well. [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) is the acceptable date format. */
    validUntilDateBefore?: Date;
    /** Date to filter Bundles having their `valid_until_date` before or after the specified date. Can be `ValidUntilDate>=` or `ValidUntilDate<=`. Both can be used in conjunction as well. [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) is the acceptable date format. */
    validUntilDateAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: BundleInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface BundleListInstanceOptions {
    /** The verification status of the Bundle resource. Please refer to [Bundle Statuses](https://www.twilio.com/docs/phone-numbers/regulatory/api/bundles#bundle-statuses) for more details. */
    status?: BundleStatus;
    /** The string that you assigned to describe the resource. The column can contain 255 variable characters. */
    friendlyName?: string;
    /** The unique string of a [Regulation resource](https://www.twilio.com/docs/phone-numbers/regulatory/api/regulations) that is associated to the Bundle resource. */
    regulationSid?: string;
    /** The 2-digit [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the Bundle\'s phone number country ownership request. */
    isoCountry?: string;
    /** The type of phone number of the Bundle\'s ownership request. Can be `local`, `mobile`, `national`, or `tollfree`. */
    numberType?: string;
    /** Indicates that the Bundle is a valid Bundle until a specified expiration date. */
    hasValidUntilDate?: boolean;
    /** Can be `valid-until` or `date-updated`. Defaults to `date-created`. */
    sortBy?: BundleSortBy;
    /** Default is `DESC`. Can be `ASC` or `DESC`. */
    sortDirection?: BundleSortDirection;
    /** Date to filter Bundles having their `valid_until_date` before or after the specified date. Can be `ValidUntilDate>=` or `ValidUntilDate<=`. Both can be used in conjunction as well. [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) is the acceptable date format. */
    validUntilDate?: Date;
    /** Date to filter Bundles having their `valid_until_date` before or after the specified date. Can be `ValidUntilDate>=` or `ValidUntilDate<=`. Both can be used in conjunction as well. [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) is the acceptable date format. */
    validUntilDateBefore?: Date;
    /** Date to filter Bundles having their `valid_until_date` before or after the specified date. Can be `ValidUntilDate>=` or `ValidUntilDate<=`. Both can be used in conjunction as well. [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) is the acceptable date format. */
    validUntilDateAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface BundleListInstancePageOptions {
    /** The verification status of the Bundle resource. Please refer to [Bundle Statuses](https://www.twilio.com/docs/phone-numbers/regulatory/api/bundles#bundle-statuses) for more details. */
    status?: BundleStatus;
    /** The string that you assigned to describe the resource. The column can contain 255 variable characters. */
    friendlyName?: string;
    /** The unique string of a [Regulation resource](https://www.twilio.com/docs/phone-numbers/regulatory/api/regulations) that is associated to the Bundle resource. */
    regulationSid?: string;
    /** The 2-digit [ISO country code](https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2) of the Bundle\'s phone number country ownership request. */
    isoCountry?: string;
    /** The type of phone number of the Bundle\'s ownership request. Can be `local`, `mobile`, `national`, or `tollfree`. */
    numberType?: string;
    /** Indicates that the Bundle is a valid Bundle until a specified expiration date. */
    hasValidUntilDate?: boolean;
    /** Can be `valid-until` or `date-updated`. Defaults to `date-created`. */
    sortBy?: BundleSortBy;
    /** Default is `DESC`. Can be `ASC` or `DESC`. */
    sortDirection?: BundleSortDirection;
    /** Date to filter Bundles having their `valid_until_date` before or after the specified date. Can be `ValidUntilDate>=` or `ValidUntilDate<=`. Both can be used in conjunction as well. [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) is the acceptable date format. */
    validUntilDate?: Date;
    /** Date to filter Bundles having their `valid_until_date` before or after the specified date. Can be `ValidUntilDate>=` or `ValidUntilDate<=`. Both can be used in conjunction as well. [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) is the acceptable date format. */
    validUntilDateBefore?: Date;
    /** Date to filter Bundles having their `valid_until_date` before or after the specified date. Can be `ValidUntilDate>=` or `ValidUntilDate<=`. Both can be used in conjunction as well. [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) is the acceptable date format. */
    validUntilDateAfter?: Date;
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface BundleContext {
    bundleCopies: BundleCopyListInstance;
    evaluations: EvaluationListInstance;
    itemAssignments: ItemAssignmentListInstance;
    replaceItems: ReplaceItemsListInstance;
    /**
     * Remove a BundleInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a BundleInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BundleInstance
     */
    fetch(callback?: (error: Error | null, item?: BundleInstance) => any): Promise<BundleInstance>;
    /**
     * Update a BundleInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BundleInstance
     */
    update(callback?: (error: Error | null, item?: BundleInstance) => any): Promise<BundleInstance>;
    /**
     * Update a BundleInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BundleInstance
     */
    update(params: BundleContextUpdateOptions, callback?: (error: Error | null, item?: BundleInstance) => any): Promise<BundleInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface BundleContextSolution {
    sid: string;
}
export declare class BundleContextImpl implements BundleContext {
    protected _version: V2;
    protected _solution: BundleContextSolution;
    protected _uri: string;
    protected _bundleCopies?: BundleCopyListInstance;
    protected _evaluations?: EvaluationListInstance;
    protected _itemAssignments?: ItemAssignmentListInstance;
    protected _replaceItems?: ReplaceItemsListInstance;
    constructor(_version: V2, sid: string);
    get bundleCopies(): BundleCopyListInstance;
    get evaluations(): EvaluationListInstance;
    get itemAssignments(): ItemAssignmentListInstance;
    get replaceItems(): ReplaceItemsListInstance;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    fetch(callback?: (error: Error | null, item?: BundleInstance) => any): Promise<BundleInstance>;
    update(params?: BundleContextUpdateOptions | ((error: Error | null, item?: BundleInstance) => any), callback?: (error: Error | null, item?: BundleInstance) => any): Promise<BundleInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): BundleContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface BundlePayload extends TwilioResponsePayload {
    results: BundleResource[];
}
interface BundleResource {
    sid: string;
    account_sid: string;
    regulation_sid: string;
    friendly_name: string;
    status: BundleStatus;
    valid_until: Date;
    email: string;
    status_callback: string;
    date_created: Date;
    date_updated: Date;
    url: string;
    links: Record<string, string>;
}
export declare class BundleInstance {
    protected _version: V2;
    protected _solution: BundleContextSolution;
    protected _context?: BundleContext;
    constructor(_version: V2, payload: BundleResource, sid?: string);
    /**
     * The unique string that we created to identify the Bundle resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Bundle resource.
     */
    accountSid: string;
    /**
     * The unique string of a regulation that is associated to the Bundle resource.
     */
    regulationSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    status: BundleStatus;
    /**
     * The date and time in GMT in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format when the resource will be valid until.
     */
    validUntil: Date;
    /**
     * The email address that will receive updates when the Bundle resource changes status.
     */
    email: string;
    /**
     * The URL we call to inform your application of status changes.
     */
    statusCallback: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The absolute URL of the Bundle resource.
     */
    url: string;
    /**
     * The URLs of the Assigned Items of the Bundle resource.
     */
    links: Record<string, string>;
    private get _proxy();
    /**
     * Remove a BundleInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Fetch a BundleInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BundleInstance
     */
    fetch(callback?: (error: Error | null, item?: BundleInstance) => any): Promise<BundleInstance>;
    /**
     * Update a BundleInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BundleInstance
     */
    update(callback?: (error: Error | null, item?: BundleInstance) => any): Promise<BundleInstance>;
    /**
     * Update a BundleInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BundleInstance
     */
    update(params: BundleContextUpdateOptions, callback?: (error: Error | null, item?: BundleInstance) => any): Promise<BundleInstance>;
    /**
     * Access the bundleCopies.
     */
    bundleCopies(): BundleCopyListInstance;
    /**
     * Access the evaluations.
     */
    evaluations(): EvaluationListInstance;
    /**
     * Access the itemAssignments.
     */
    itemAssignments(): ItemAssignmentListInstance;
    /**
     * Access the replaceItems.
     */
    replaceItems(): ReplaceItemsListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        regulationSid: string;
        friendlyName: string;
        status: BundleStatus;
        validUntil: Date;
        email: string;
        statusCallback: string;
        dateCreated: Date;
        dateUpdated: Date;
        url: string;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface BundleSolution {
}
export interface BundleListInstance {
    _version: V2;
    _solution: BundleSolution;
    _uri: string;
    (sid: string): BundleContext;
    get(sid: string): BundleContext;
    /**
     * Create a BundleInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BundleInstance
     */
    create(params: BundleListInstanceCreateOptions, callback?: (error: Error | null, item?: BundleInstance) => any): Promise<BundleInstance>;
    /**
     * Streams BundleInstance records from the API.
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
     * @param { BundleListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: BundleInstance, done: (err?: Error) => void) => void): void;
    each(params: BundleListInstanceEachOptions, callback?: (item: BundleInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of BundleInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: BundlePage) => any): Promise<BundlePage>;
    /**
     * Lists BundleInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BundleListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: BundleInstance[]) => any): Promise<BundleInstance[]>;
    list(params: BundleListInstanceOptions, callback?: (error: Error | null, items: BundleInstance[]) => any): Promise<BundleInstance[]>;
    /**
     * Retrieve a single page of BundleInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BundleListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: BundlePage) => any): Promise<BundlePage>;
    page(params: BundleListInstancePageOptions, callback?: (error: Error | null, items: BundlePage) => any): Promise<BundlePage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BundleListInstance(version: V2): BundleListInstance;
export declare class BundlePage extends Page<V2, BundlePayload, BundleResource, BundleInstance> {
    /**
     * Initialize the BundlePage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V2, response: Response<string>, solution: BundleSolution);
    /**
     * Build an instance of BundleInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: BundleResource): BundleInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
