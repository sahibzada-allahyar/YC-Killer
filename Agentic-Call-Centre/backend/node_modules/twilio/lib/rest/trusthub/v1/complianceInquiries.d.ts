/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to update a ComplianceInquiriesInstance
 */
export interface ComplianceInquiriesContextUpdateOptions {
    /** The unique SID identifier of the Primary Customer Profile that should be used as a parent. Only necessary when creating a secondary Customer Profile. */
    primaryProfileSid: string;
}
/**
 * Options to pass to create a ComplianceInquiriesInstance
 */
export interface ComplianceInquiriesListInstanceCreateOptions {
    /** The unique SID identifier of the Primary Customer Profile that should be used as a parent. Only necessary when creating a secondary Customer Profile. */
    primaryProfileSid: string;
    /** The email address that approval status updates will be sent to. If not specified, the email address associated with your primary customer profile will be used. */
    notificationEmail?: string;
}
export interface ComplianceInquiriesContext {
    /**
     * Update a ComplianceInquiriesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ComplianceInquiriesInstance
     */
    update(params: ComplianceInquiriesContextUpdateOptions, callback?: (error: Error | null, item?: ComplianceInquiriesInstance) => any): Promise<ComplianceInquiriesInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface ComplianceInquiriesContextSolution {
    customerId: string;
}
export declare class ComplianceInquiriesContextImpl implements ComplianceInquiriesContext {
    protected _version: V1;
    protected _solution: ComplianceInquiriesContextSolution;
    protected _uri: string;
    constructor(_version: V1, customerId: string);
    update(params: ComplianceInquiriesContextUpdateOptions, callback?: (error: Error | null, item?: ComplianceInquiriesInstance) => any): Promise<ComplianceInquiriesInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): ComplianceInquiriesContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface ComplianceInquiriesResource {
    inquiry_id: string;
    inquiry_session_token: string;
    customer_id: string;
    url: string;
}
export declare class ComplianceInquiriesInstance {
    protected _version: V1;
    protected _solution: ComplianceInquiriesContextSolution;
    protected _context?: ComplianceInquiriesContext;
    constructor(_version: V1, payload: ComplianceInquiriesResource, customerId?: string);
    /**
     * The unique ID used to start an embedded compliance registration session.
     */
    inquiryId: string;
    /**
     * The session token used to start an embedded compliance registration session.
     */
    inquirySessionToken: string;
    /**
     * The CustomerID matching the Customer Profile that should be resumed or resubmitted for editing.
     */
    customerId: string;
    /**
     * The URL of this resource.
     */
    url: string;
    private get _proxy();
    /**
     * Update a ComplianceInquiriesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ComplianceInquiriesInstance
     */
    update(params: ComplianceInquiriesContextUpdateOptions, callback?: (error: Error | null, item?: ComplianceInquiriesInstance) => any): Promise<ComplianceInquiriesInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        inquiryId: string;
        inquirySessionToken: string;
        customerId: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface ComplianceInquiriesSolution {
}
export interface ComplianceInquiriesListInstance {
    _version: V1;
    _solution: ComplianceInquiriesSolution;
    _uri: string;
    (customerId: string): ComplianceInquiriesContext;
    get(customerId: string): ComplianceInquiriesContext;
    /**
     * Create a ComplianceInquiriesInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ComplianceInquiriesInstance
     */
    create(params: ComplianceInquiriesListInstanceCreateOptions, callback?: (error: Error | null, item?: ComplianceInquiriesInstance) => any): Promise<ComplianceInquiriesInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ComplianceInquiriesListInstance(version: V1): ComplianceInquiriesListInstance;
export {};
