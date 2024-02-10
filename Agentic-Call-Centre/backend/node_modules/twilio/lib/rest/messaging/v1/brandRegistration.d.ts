/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import Page, { TwilioResponsePayload } from "../../../base/Page";
import Response from "../../../http/response";
import V1 from "../V1";
import { BrandRegistrationOtpListInstance } from "./brandRegistration/brandRegistrationOtp";
import { BrandVettingListInstance } from "./brandRegistration/brandVetting";
export type BrandRegistrationBrandFeedback = "TAX_ID" | "STOCK_SYMBOL" | "NONPROFIT" | "GOVERNMENT_ENTITY" | "OTHERS";
export type BrandRegistrationIdentityStatus = "SELF_DECLARED" | "UNVERIFIED" | "VERIFIED" | "VETTED_VERIFIED";
export type BrandRegistrationStatus = "PENDING" | "APPROVED" | "FAILED" | "IN_REVIEW" | "DELETED";
/**
 * Options to pass to create a BrandRegistrationInstance
 */
export interface BrandRegistrationListInstanceCreateOptions {
    /** Customer Profile Bundle Sid. */
    customerProfileBundleSid: string;
    /** A2P Messaging Profile Bundle Sid. */
    a2PProfileBundleSid: string;
    /** Type of brand being created. One of: \\\"STANDARD\\\", \\\"SOLE_PROPRIETOR\\\". SOLE_PROPRIETOR is for low volume, SOLE_PROPRIETOR use cases. STANDARD is for all other use cases. */
    brandType?: string;
    /** A boolean that specifies whether brand should be a mock or not. If true, brand will be registered as a mock brand. Defaults to false if no value is provided. */
    mock?: boolean;
    /** A flag to disable automatic secondary vetting for brands which it would otherwise be done. */
    skipAutomaticSecVet?: boolean;
}
/**
 * Options to pass to each
 */
export interface BrandRegistrationListInstanceEachOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Function to process each record. If this and a positional callback are passed, this one will be used */
    callback?: (item: BrandRegistrationInstance, done: (err?: Error) => void) => void;
    /** Function to be called upon completion of streaming */
    done?: Function;
    /** Upper limit for the number of records to return. each() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to list
 */
export interface BrandRegistrationListInstanceOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Upper limit for the number of records to return. list() guarantees never to return more than limit. Default is no limit */
    limit?: number;
}
/**
 * Options to pass to page
 */
export interface BrandRegistrationListInstancePageOptions {
    /** How many resources to return in each list page. The default is 50, and the maximum is 1000. */
    pageSize?: number;
    /** Page Number, this value is simply for client state */
    pageNumber?: number;
    /** PageToken provided by the API */
    pageToken?: string;
}
export interface BrandRegistrationContext {
    brandRegistrationOtps: BrandRegistrationOtpListInstance;
    brandVettings: BrandVettingListInstance;
    /**
     * Fetch a BrandRegistrationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BrandRegistrationInstance
     */
    fetch(callback?: (error: Error | null, item?: BrandRegistrationInstance) => any): Promise<BrandRegistrationInstance>;
    /**
     * Update a BrandRegistrationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BrandRegistrationInstance
     */
    update(callback?: (error: Error | null, item?: BrandRegistrationInstance) => any): Promise<BrandRegistrationInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface BrandRegistrationContextSolution {
    sid: string;
}
export declare class BrandRegistrationContextImpl implements BrandRegistrationContext {
    protected _version: V1;
    protected _solution: BrandRegistrationContextSolution;
    protected _uri: string;
    protected _brandRegistrationOtps?: BrandRegistrationOtpListInstance;
    protected _brandVettings?: BrandVettingListInstance;
    constructor(_version: V1, sid: string);
    get brandRegistrationOtps(): BrandRegistrationOtpListInstance;
    get brandVettings(): BrandVettingListInstance;
    fetch(callback?: (error: Error | null, item?: BrandRegistrationInstance) => any): Promise<BrandRegistrationInstance>;
    update(callback?: (error: Error | null, item?: BrandRegistrationInstance) => any): Promise<BrandRegistrationInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): BrandRegistrationContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface BrandRegistrationPayload extends TwilioResponsePayload {
    data: BrandRegistrationResource[];
}
interface BrandRegistrationResource {
    sid: string;
    account_sid: string;
    customer_profile_bundle_sid: string;
    a2p_profile_bundle_sid: string;
    date_created: Date;
    date_updated: Date;
    brand_type: string;
    status: BrandRegistrationStatus;
    tcr_id: string;
    failure_reason: string;
    url: string;
    brand_score: number;
    brand_feedback: Array<BrandRegistrationBrandFeedback>;
    identity_status: BrandRegistrationIdentityStatus;
    russell_3000: boolean;
    government_entity: boolean;
    tax_exempt_status: string;
    skip_automatic_sec_vet: boolean;
    mock: boolean;
    links: Record<string, string>;
}
export declare class BrandRegistrationInstance {
    protected _version: V1;
    protected _solution: BrandRegistrationContextSolution;
    protected _context?: BrandRegistrationContext;
    constructor(_version: V1, payload: BrandRegistrationResource, sid?: string);
    /**
     * The unique string to identify Brand Registration.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Brand Registration resource.
     */
    accountSid: string;
    /**
     * A2P Messaging Profile Bundle BundleSid.
     */
    customerProfileBundleSid: string;
    /**
     * A2P Messaging Profile Bundle BundleSid.
     */
    a2pProfileBundleSid: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * Type of brand. One of: \"STANDARD\", \"SOLE_PROPRIETOR\". SOLE_PROPRIETOR is for the low volume, SOLE_PROPRIETOR campaign use case. There can only be one SOLE_PROPRIETOR campaign created per SOLE_PROPRIETOR brand. STANDARD is for all other campaign use cases. Multiple campaign use cases can be created per STANDARD brand.
     */
    brandType: string;
    status: BrandRegistrationStatus;
    /**
     * Campaign Registry (TCR) Brand ID. Assigned only after successful brand registration.
     */
    tcrId: string;
    /**
     * A reason why brand registration has failed. Only applicable when status is FAILED.
     */
    failureReason: string;
    /**
     * The absolute URL of the Brand Registration resource.
     */
    url: string;
    /**
     * The secondary vetting score if it was done. Otherwise, it will be the brand score if it\'s returned from TCR. It may be null if no score is available.
     */
    brandScore: number;
    /**
     * Feedback on how to improve brand score
     */
    brandFeedback: Array<BrandRegistrationBrandFeedback>;
    identityStatus: BrandRegistrationIdentityStatus;
    /**
     * Publicly traded company identified in the Russell 3000 Index
     */
    russell3000: boolean;
    /**
     * Identified as a government entity
     */
    governmentEntity: boolean;
    /**
     * Nonprofit organization tax-exempt status per section 501 of the U.S. tax code.
     */
    taxExemptStatus: string;
    /**
     * A flag to disable automatic secondary vetting for brands which it would otherwise be done.
     */
    skipAutomaticSecVet: boolean;
    /**
     * A boolean that specifies whether brand should be a mock or not. If true, brand will be registered as a mock brand. Defaults to false if no value is provided.
     */
    mock: boolean;
    links: Record<string, string>;
    private get _proxy();
    /**
     * Fetch a BrandRegistrationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BrandRegistrationInstance
     */
    fetch(callback?: (error: Error | null, item?: BrandRegistrationInstance) => any): Promise<BrandRegistrationInstance>;
    /**
     * Update a BrandRegistrationInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BrandRegistrationInstance
     */
    update(callback?: (error: Error | null, item?: BrandRegistrationInstance) => any): Promise<BrandRegistrationInstance>;
    /**
     * Access the brandRegistrationOtps.
     */
    brandRegistrationOtps(): BrandRegistrationOtpListInstance;
    /**
     * Access the brandVettings.
     */
    brandVettings(): BrandVettingListInstance;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        customerProfileBundleSid: string;
        a2pProfileBundleSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        brandType: string;
        status: BrandRegistrationStatus;
        tcrId: string;
        failureReason: string;
        url: string;
        brandScore: number;
        brandFeedback: BrandRegistrationBrandFeedback[];
        identityStatus: BrandRegistrationIdentityStatus;
        russell3000: boolean;
        governmentEntity: boolean;
        taxExemptStatus: string;
        skipAutomaticSecVet: boolean;
        mock: boolean;
        links: Record<string, string>;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface BrandRegistrationSolution {
}
export interface BrandRegistrationListInstance {
    _version: V1;
    _solution: BrandRegistrationSolution;
    _uri: string;
    (sid: string): BrandRegistrationContext;
    get(sid: string): BrandRegistrationContext;
    /**
     * Create a BrandRegistrationInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BrandRegistrationInstance
     */
    create(params: BrandRegistrationListInstanceCreateOptions, callback?: (error: Error | null, item?: BrandRegistrationInstance) => any): Promise<BrandRegistrationInstance>;
    /**
     * Streams BrandRegistrationInstance records from the API.
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
     * @param { BrandRegistrationListInstanceEachOptions } [params] - Options for request
     * @param { function } [callback] - Function to process each record
     */
    each(callback?: (item: BrandRegistrationInstance, done: (err?: Error) => void) => void): void;
    each(params: BrandRegistrationListInstanceEachOptions, callback?: (item: BrandRegistrationInstance, done: (err?: Error) => void) => void): void;
    /**
     * Retrieve a single target page of BrandRegistrationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * @param { string } [targetUrl] - API-generated URL for the requested results page
     * @param { function } [callback] - Callback to handle list of records
     */
    getPage(targetUrl: string, callback?: (error: Error | null, items: BrandRegistrationPage) => any): Promise<BrandRegistrationPage>;
    /**
     * Lists BrandRegistrationInstance records from the API as a list.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BrandRegistrationListInstanceOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    list(callback?: (error: Error | null, items: BrandRegistrationInstance[]) => any): Promise<BrandRegistrationInstance[]>;
    list(params: BrandRegistrationListInstanceOptions, callback?: (error: Error | null, items: BrandRegistrationInstance[]) => any): Promise<BrandRegistrationInstance[]>;
    /**
     * Retrieve a single page of BrandRegistrationInstance records from the API.
     *
     * The request is executed immediately.
     *
     * If a function is passed as the first argument, it will be used as the callback
     * function.
     *
     * @param { BrandRegistrationListInstancePageOptions } [params] - Options for request
     * @param { function } [callback] - Callback to handle list of records
     */
    page(callback?: (error: Error | null, items: BrandRegistrationPage) => any): Promise<BrandRegistrationPage>;
    page(params: BrandRegistrationListInstancePageOptions, callback?: (error: Error | null, items: BrandRegistrationPage) => any): Promise<BrandRegistrationPage>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BrandRegistrationListInstance(version: V1): BrandRegistrationListInstance;
export declare class BrandRegistrationPage extends Page<V1, BrandRegistrationPayload, BrandRegistrationResource, BrandRegistrationInstance> {
    /**
     * Initialize the BrandRegistrationPage
     *
     * @param version - Version of the resource
     * @param response - Response from the API
     * @param solution - Path solution
     */
    constructor(version: V1, response: Response<string>, solution: BrandRegistrationSolution);
    /**
     * Build an instance of BrandRegistrationInstance
     *
     * @param payload - Payload response from the API
     */
    getInstance(payload: BrandRegistrationResource): BrandRegistrationInstance;
    [inspect.custom](depth: any, options: InspectOptions): string;
}
export {};
