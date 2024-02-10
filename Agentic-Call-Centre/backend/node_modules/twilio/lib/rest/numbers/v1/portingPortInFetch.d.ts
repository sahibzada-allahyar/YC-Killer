/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export interface PortingPortInFetchContext {
    /**
     * Fetch a PortingPortInFetchInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PortingPortInFetchInstance
     */
    fetch(callback?: (error: Error | null, item?: PortingPortInFetchInstance) => any): Promise<PortingPortInFetchInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PortingPortInFetchContextSolution {
    portInRequestSid: string;
}
export declare class PortingPortInFetchContextImpl implements PortingPortInFetchContext {
    protected _version: V1;
    protected _solution: PortingPortInFetchContextSolution;
    protected _uri: string;
    constructor(_version: V1, portInRequestSid: string);
    fetch(callback?: (error: Error | null, item?: PortingPortInFetchInstance) => any): Promise<PortingPortInFetchInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PortingPortInFetchContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PortingPortInFetchResource {
    port_in_request_sid: string;
    url: string;
    account_sid: string;
    notification_emails: Array<string>;
    target_port_in_date: Date;
    target_port_in_time_range_start: string;
    target_port_in_time_range_end: string;
    losing_carrier_information: any;
    phone_numbers: Array<any>;
    documents: Array<string>;
}
export declare class PortingPortInFetchInstance {
    protected _version: V1;
    protected _solution: PortingPortInFetchContextSolution;
    protected _context?: PortingPortInFetchContext;
    constructor(_version: V1, payload: PortingPortInFetchResource, portInRequestSid?: string);
    /**
     * The SID of the Port In request. This is a unique identifier of the port in request.
     */
    portInRequestSid: string;
    /**
     * The URL of this Port In request
     */
    url: string;
    /**
     * The Account SID that the numbers will be added to after they are ported into Twilio.
     */
    accountSid: string;
    /**
     * List of emails for getting notifications about the LOA signing process. Allowed Max 10 emails.
     */
    notificationEmails: Array<string>;
    /**
     * Minimum number of days in the future (at least 2 days) needs to be established with the Ops team for validation.
     */
    targetPortInDate: Date;
    /**
     * Minimum hour in the future needs to be established with the Ops team for validation.
     */
    targetPortInTimeRangeStart: string;
    /**
     * Maximum hour in the future needs to be established with the Ops team for validation.
     */
    targetPortInTimeRangeEnd: string;
    /**
     * The information for the losing carrier.
     */
    losingCarrierInformation: any;
    /**
     * The list of phone numbers to Port in. Phone numbers are in E.164 format (e.g. +16175551212).
     */
    phoneNumbers: Array<any>;
    /**
     * The list of documents SID referencing a utility bills
     */
    documents: Array<string>;
    private get _proxy();
    /**
     * Fetch a PortingPortInFetchInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PortingPortInFetchInstance
     */
    fetch(callback?: (error: Error | null, item?: PortingPortInFetchInstance) => any): Promise<PortingPortInFetchInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        portInRequestSid: string;
        url: string;
        accountSid: string;
        notificationEmails: string[];
        targetPortInDate: Date;
        targetPortInTimeRangeStart: string;
        targetPortInTimeRangeEnd: string;
        losingCarrierInformation: any;
        phoneNumbers: any[];
        documents: string[];
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PortingPortInFetchSolution {
}
export interface PortingPortInFetchListInstance {
    _version: V1;
    _solution: PortingPortInFetchSolution;
    _uri: string;
    (portInRequestSid: string): PortingPortInFetchContext;
    get(portInRequestSid: string): PortingPortInFetchContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PortingPortInFetchListInstance(version: V1): PortingPortInFetchListInstance;
export {};
