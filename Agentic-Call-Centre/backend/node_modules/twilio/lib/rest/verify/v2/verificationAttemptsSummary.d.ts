/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../V2";
export type VerificationAttemptsSummaryChannels = "sms" | "call" | "email" | "whatsapp";
/**
 * Options to pass to fetch a VerificationAttemptsSummaryInstance
 */
export interface VerificationAttemptsSummaryContextFetchOptions {
    /** Filter used to consider only Verification Attempts of the given verify service on the summary aggregation. */
    verifyServiceSid?: string;
    /** Datetime filter used to consider only Verification Attempts created after this datetime on the summary aggregation. Given as GMT in ISO 8601 formatted datetime string: yyyy-MM-dd\'T\'HH:mm:ss\'Z. */
    dateCreatedAfter?: Date;
    /** Datetime filter used to consider only Verification Attempts created before this datetime on the summary aggregation. Given as GMT in ISO 8601 formatted datetime string: yyyy-MM-dd\'T\'HH:mm:ss\'Z. */
    dateCreatedBefore?: Date;
    /** Filter used to consider only Verification Attempts sent to the specified destination country on the summary aggregation. */
    country?: string;
    /** Filter Verification Attempts considered on the summary aggregation by communication channel. Valid values are `SMS`, `CALL` and `WHATSAPP` */
    channel?: VerificationAttemptsSummaryChannels;
    /** Filter the Verification Attempts considered on the summary aggregation by Destination prefix. It is the prefix of a phone number in E.164 format. */
    destinationPrefix?: string;
}
export interface VerificationAttemptsSummaryContext {
    /**
     * Fetch a VerificationAttemptsSummaryInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationAttemptsSummaryInstance
     */
    fetch(callback?: (error: Error | null, item?: VerificationAttemptsSummaryInstance) => any): Promise<VerificationAttemptsSummaryInstance>;
    /**
     * Fetch a VerificationAttemptsSummaryInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationAttemptsSummaryInstance
     */
    fetch(params: VerificationAttemptsSummaryContextFetchOptions, callback?: (error: Error | null, item?: VerificationAttemptsSummaryInstance) => any): Promise<VerificationAttemptsSummaryInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface VerificationAttemptsSummaryContextSolution {
}
export declare class VerificationAttemptsSummaryContextImpl implements VerificationAttemptsSummaryContext {
    protected _version: V2;
    protected _solution: VerificationAttemptsSummaryContextSolution;
    protected _uri: string;
    constructor(_version: V2);
    fetch(params?: VerificationAttemptsSummaryContextFetchOptions | ((error: Error | null, item?: VerificationAttemptsSummaryInstance) => any), callback?: (error: Error | null, item?: VerificationAttemptsSummaryInstance) => any): Promise<VerificationAttemptsSummaryInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): VerificationAttemptsSummaryContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface VerificationAttemptsSummaryResource {
    total_attempts: number;
    total_converted: number;
    total_unconverted: number;
    conversion_rate_percentage: number;
    url: string;
}
export declare class VerificationAttemptsSummaryInstance {
    protected _version: V2;
    protected _solution: VerificationAttemptsSummaryContextSolution;
    protected _context?: VerificationAttemptsSummaryContext;
    constructor(_version: V2, payload: VerificationAttemptsSummaryResource);
    /**
     * Total of attempts made according to the provided filters
     */
    totalAttempts: number;
    /**
     * Total of  attempts made that were confirmed by the end user, according to the provided filters.
     */
    totalConverted: number;
    /**
     * Total of attempts made that were not confirmed by the end user, according to the provided filters.
     */
    totalUnconverted: number;
    /**
     * Percentage of the confirmed messages over the total, defined by (total_converted/total_attempts)*100.
     */
    conversionRatePercentage: number;
    url: string;
    private get _proxy();
    /**
     * Fetch a VerificationAttemptsSummaryInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationAttemptsSummaryInstance
     */
    fetch(callback?: (error: Error | null, item?: VerificationAttemptsSummaryInstance) => any): Promise<VerificationAttemptsSummaryInstance>;
    /**
     * Fetch a VerificationAttemptsSummaryInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationAttemptsSummaryInstance
     */
    fetch(params: VerificationAttemptsSummaryContextFetchOptions, callback?: (error: Error | null, item?: VerificationAttemptsSummaryInstance) => any): Promise<VerificationAttemptsSummaryInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        totalAttempts: number;
        totalConverted: number;
        totalUnconverted: number;
        conversionRatePercentage: number;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface VerificationAttemptsSummarySolution {
}
export interface VerificationAttemptsSummaryListInstance {
    _version: V2;
    _solution: VerificationAttemptsSummarySolution;
    _uri: string;
    (): VerificationAttemptsSummaryContext;
    get(): VerificationAttemptsSummaryContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function VerificationAttemptsSummaryListInstance(version: V2): VerificationAttemptsSummaryListInstance;
export {};
