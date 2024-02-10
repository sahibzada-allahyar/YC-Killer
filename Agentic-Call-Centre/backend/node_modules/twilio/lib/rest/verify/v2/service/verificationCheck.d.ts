/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../../V2";
export type VerificationCheckChannel = "sms" | "call" | "email" | "whatsapp" | "sna";
/**
 * Options to pass to create a VerificationCheckInstance
 */
export interface VerificationCheckListInstanceCreateOptions {
    /** The 4-10 character string being verified. */
    code?: string;
    /** The phone number or [email](https://www.twilio.com/docs/verify/email) to verify. Either this parameter or the `verification_sid` must be specified. Phone numbers must be in [E.164 format](https://www.twilio.com/docs/glossary/what-e164). */
    to?: string;
    /** A SID that uniquely identifies the Verification Check. Either this parameter or the `to` phone number/[email](https://www.twilio.com/docs/verify/email) must be specified. */
    verificationSid?: string;
    /** The amount of the associated PSD2 compliant transaction. Requires the PSD2 Service flag enabled. */
    amount?: string;
    /** The payee of the associated PSD2 compliant transaction. Requires the PSD2 Service flag enabled. */
    payee?: string;
}
export interface VerificationCheckSolution {
    serviceSid: string;
}
export interface VerificationCheckListInstance {
    _version: V2;
    _solution: VerificationCheckSolution;
    _uri: string;
    /**
     * Create a VerificationCheckInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationCheckInstance
     */
    create(callback?: (error: Error | null, item?: VerificationCheckInstance) => any): Promise<VerificationCheckInstance>;
    /**
     * Create a VerificationCheckInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed VerificationCheckInstance
     */
    create(params: VerificationCheckListInstanceCreateOptions, callback?: (error: Error | null, item?: VerificationCheckInstance) => any): Promise<VerificationCheckInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function VerificationCheckListInstance(version: V2, serviceSid: string): VerificationCheckListInstance;
interface VerificationCheckResource {
    sid: string;
    service_sid: string;
    account_sid: string;
    to: string;
    channel: VerificationCheckChannel;
    status: string;
    valid: boolean;
    amount: string;
    payee: string;
    date_created: Date;
    date_updated: Date;
    sna_attempts_error_codes: Array<any>;
}
export declare class VerificationCheckInstance {
    protected _version: V2;
    constructor(_version: V2, payload: VerificationCheckResource, serviceSid: string);
    /**
     * The unique string that we created to identify the VerificationCheck resource.
     */
    sid: string;
    /**
     * The SID of the [Service](https://www.twilio.com/docs/verify/api/service) the resource is associated with.
     */
    serviceSid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the VerificationCheck resource.
     */
    accountSid: string;
    /**
     * The phone number or [email](https://www.twilio.com/docs/verify/email) being verified. Phone numbers must be in [E.164 format](https://www.twilio.com/docs/glossary/what-e164).
     */
    to: string;
    channel: VerificationCheckChannel;
    /**
     * The status of the verification. Can be: `pending`, `approved`, or `canceled`.
     */
    status: string;
    /**
     * Use \"status\" instead. Legacy property indicating whether the verification was successful.
     */
    valid: boolean;
    /**
     * The amount of the associated PSD2 compliant transaction. Requires the PSD2 Service flag enabled.
     */
    amount: string;
    /**
     * The payee of the associated PSD2 compliant transaction. Requires the PSD2 Service flag enabled.
     */
    payee: string;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the Verification Check resource was created.
     */
    dateCreated: Date;
    /**
     * The [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) date and time in GMT when the Verification Check resource was last updated.
     */
    dateUpdated: Date;
    /**
     * List of error codes as a result of attempting a verification using the `sna` channel. The error codes are chronologically ordered, from the first attempt to the latest attempt. This will be an empty list if no errors occured or `null` if the last channel used wasn\'t `sna`.
     */
    snaAttemptsErrorCodes: Array<any>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        serviceSid: string;
        accountSid: string;
        to: string;
        channel: VerificationCheckChannel;
        status: string;
        valid: boolean;
        amount: string;
        payee: string;
        dateCreated: Date;
        dateUpdated: Date;
        snaAttemptsErrorCodes: any[];
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
