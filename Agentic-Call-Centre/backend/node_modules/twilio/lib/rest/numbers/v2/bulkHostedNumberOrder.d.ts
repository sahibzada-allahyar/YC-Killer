/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../V2";
export type BulkHostedNumberOrderRequestStatus = "QUEUED" | "IN_PROGRESS" | "PROCESSED";
/**
 * Options to pass to fetch a BulkHostedNumberOrderInstance
 */
export interface BulkHostedNumberOrderContextFetchOptions {
    /** Order status can be used for filtering on Hosted Number Order status values. To see a complete list of order statuses, please check \'https://www.twilio.com/docs/phone-numbers/hosted-numbers/hosted-numbers-api/hosted-number-order-resource#status-values\'. */
    orderStatus?: string;
}
export interface BulkHostedNumberOrderContext {
    /**
     * Fetch a BulkHostedNumberOrderInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BulkHostedNumberOrderInstance
     */
    fetch(callback?: (error: Error | null, item?: BulkHostedNumberOrderInstance) => any): Promise<BulkHostedNumberOrderInstance>;
    /**
     * Fetch a BulkHostedNumberOrderInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BulkHostedNumberOrderInstance
     */
    fetch(params: BulkHostedNumberOrderContextFetchOptions, callback?: (error: Error | null, item?: BulkHostedNumberOrderInstance) => any): Promise<BulkHostedNumberOrderInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface BulkHostedNumberOrderContextSolution {
    bulkHostingSid: string;
}
export declare class BulkHostedNumberOrderContextImpl implements BulkHostedNumberOrderContext {
    protected _version: V2;
    protected _solution: BulkHostedNumberOrderContextSolution;
    protected _uri: string;
    constructor(_version: V2, bulkHostingSid: string);
    fetch(params?: BulkHostedNumberOrderContextFetchOptions | ((error: Error | null, item?: BulkHostedNumberOrderInstance) => any), callback?: (error: Error | null, item?: BulkHostedNumberOrderInstance) => any): Promise<BulkHostedNumberOrderInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): BulkHostedNumberOrderContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface BulkHostedNumberOrderResource {
    bulk_hosting_sid: string;
    request_status: BulkHostedNumberOrderRequestStatus;
    friendly_name: string;
    notification_email: string;
    date_created: Date;
    date_completed: Date;
    url: string;
    total_count: number;
    results: Array<any>;
}
export declare class BulkHostedNumberOrderInstance {
    protected _version: V2;
    protected _solution: BulkHostedNumberOrderContextSolution;
    protected _context?: BulkHostedNumberOrderContext;
    constructor(_version: V2, payload: BulkHostedNumberOrderResource, bulkHostingSid?: string);
    /**
     * A 34 character string that uniquely identifies this BulkHostedNumberOrder.
     */
    bulkHostingSid: string;
    requestStatus: BulkHostedNumberOrderRequestStatus;
    /**
     * A 128 character string that is a human-readable text that describes this resource.
     */
    friendlyName: string;
    /**
     * Email address used for send notifications about this Bulk hosted number request.
     */
    notificationEmail: string;
    /**
     * The date this resource was created, given as [GMT RFC 2822](http://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date that this resource was completed, given as [GMT RFC 2822](http://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCompleted: Date;
    /**
     * The URL of this BulkHostedNumberOrder resource.
     */
    url: string;
    /**
     * The total count of phone numbers in this Bulk hosting request.
     */
    totalCount: number;
    /**
     * Contains a list of all the individual hosting orders and their information, for this Bulk request. Each result object is grouped by its order status. To see a complete list of order status, please check \'https://www.twilio.com/docs/phone-numbers/hosted-numbers/hosted-numbers-api/hosted-number-order-resource#status-values\'.
     */
    results: Array<any>;
    private get _proxy();
    /**
     * Fetch a BulkHostedNumberOrderInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BulkHostedNumberOrderInstance
     */
    fetch(callback?: (error: Error | null, item?: BulkHostedNumberOrderInstance) => any): Promise<BulkHostedNumberOrderInstance>;
    /**
     * Fetch a BulkHostedNumberOrderInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BulkHostedNumberOrderInstance
     */
    fetch(params: BulkHostedNumberOrderContextFetchOptions, callback?: (error: Error | null, item?: BulkHostedNumberOrderInstance) => any): Promise<BulkHostedNumberOrderInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        bulkHostingSid: string;
        requestStatus: BulkHostedNumberOrderRequestStatus;
        friendlyName: string;
        notificationEmail: string;
        dateCreated: Date;
        dateCompleted: Date;
        url: string;
        totalCount: number;
        results: any[];
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface BulkHostedNumberOrderSolution {
}
export interface BulkHostedNumberOrderListInstance {
    _version: V2;
    _solution: BulkHostedNumberOrderSolution;
    _uri: string;
    (bulkHostingSid: string): BulkHostedNumberOrderContext;
    get(bulkHostingSid: string): BulkHostedNumberOrderContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BulkHostedNumberOrderListInstance(version: V2): BulkHostedNumberOrderListInstance;
export {};
