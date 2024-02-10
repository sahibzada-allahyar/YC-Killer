/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export interface BulkEligibilityContext {
    /**
     * Fetch a BulkEligibilityInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BulkEligibilityInstance
     */
    fetch(callback?: (error: Error | null, item?: BulkEligibilityInstance) => any): Promise<BulkEligibilityInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface BulkEligibilityContextSolution {
    requestId: string;
}
export declare class BulkEligibilityContextImpl implements BulkEligibilityContext {
    protected _version: V1;
    protected _solution: BulkEligibilityContextSolution;
    protected _uri: string;
    constructor(_version: V1, requestId: string);
    fetch(callback?: (error: Error | null, item?: BulkEligibilityInstance) => any): Promise<BulkEligibilityInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): BulkEligibilityContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface BulkEligibilityResource {
    request_id: string;
    url: string;
    results: Array<any>;
    friendly_name: string;
    status: string;
    date_created: Date;
    date_completed: Date;
}
export declare class BulkEligibilityInstance {
    protected _version: V1;
    protected _solution: BulkEligibilityContextSolution;
    protected _context?: BulkEligibilityContext;
    constructor(_version: V1, payload: BulkEligibilityResource, requestId?: string);
    /**
     * The SID of the bulk eligibility check that you want to know about.
     */
    requestId: string;
    /**
     * This is the url of the request that you\'re trying to reach out to locate the resource.
     */
    url: string;
    /**
     * The result set that contains the eligibility check response for each requested number, each result has at least the following attributes:  phone_number: The requested phone number ,hosting_account_sid: The account sid where the phone number will be hosted, country: Phone number’s country, eligibility_status: Indicates the eligibility status of the PN (Eligible/Ineligible), eligibility_sub_status: Indicates the sub status of the eligibility , ineligibility_reason: Reason for number\'s ineligibility (if applicable), next_step: Suggested next step in the hosting process based on the eligibility status.
     */
    results: Array<any>;
    /**
     * This is the string that you assigned as a friendly name for describing the eligibility check request.
     */
    friendlyName: string;
    /**
     * This is the status of the bulk eligibility check request. (Example: COMPLETE, IN_PROGRESS)
     */
    status: string;
    dateCreated: Date;
    dateCompleted: Date;
    private get _proxy();
    /**
     * Fetch a BulkEligibilityInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BulkEligibilityInstance
     */
    fetch(callback?: (error: Error | null, item?: BulkEligibilityInstance) => any): Promise<BulkEligibilityInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        requestId: string;
        url: string;
        results: any[];
        friendlyName: string;
        status: string;
        dateCreated: Date;
        dateCompleted: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface BulkEligibilitySolution {
}
export interface BulkEligibilityListInstance {
    _version: V1;
    _solution: BulkEligibilitySolution;
    _uri: string;
    (requestId: string): BulkEligibilityContext;
    get(requestId: string): BulkEligibilityContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BulkEligibilityListInstance(version: V1): BulkEligibilityListInstance;
export {};
