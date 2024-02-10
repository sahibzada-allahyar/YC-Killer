/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2010 from "../../../V2010";
export type PaymentBankAccountType = "consumer-checking" | "consumer-savings" | "commercial-checking";
export type PaymentCapture = "payment-card-number" | "expiration-date" | "security-code" | "postal-code" | "bank-routing-number" | "bank-account-number";
export type PaymentPaymentMethod = "credit-card" | "ach-debit";
export type PaymentStatus = "complete" | "cancel";
export type PaymentTokenType = "one-time" | "reusable";
/**
 * Options to pass to update a PaymentInstance
 */
export interface PaymentContextUpdateOptions {
    /** A unique token that will be used to ensure that multiple API calls with the same information do not result in multiple transactions. This should be a unique string value per API call and can be a randomly generated. */
    idempotencyKey: string;
    /** Provide an absolute or relative URL to receive status updates regarding your Pay session. Read more about the [Update](https://www.twilio.com/docs/voice/api/payment-resource#statuscallback-update) and [Complete/Cancel](https://www.twilio.com/docs/voice/api/payment-resource#statuscallback-cancelcomplete) POST requests. */
    statusCallback: string;
    /**  */
    capture?: PaymentCapture;
    /**  */
    status?: PaymentStatus;
}
/**
 * Options to pass to create a PaymentInstance
 */
export interface PaymentListInstanceCreateOptions {
    /** A unique token that will be used to ensure that multiple API calls with the same information do not result in multiple transactions. This should be a unique string value per API call and can be a randomly generated. */
    idempotencyKey: string;
    /** Provide an absolute or relative URL to receive status updates regarding your Pay session. Read more about the [expected StatusCallback values](https://www.twilio.com/docs/voice/api/payment-resource#statuscallback) */
    statusCallback: string;
    /**  */
    bankAccountType?: PaymentBankAccountType;
    /** A positive decimal value less than 1,000,000 to charge against the credit card or bank account. Default currency can be overwritten with `currency` field. Leave blank or set to 0 to tokenize. */
    chargeAmount?: number;
    /** The currency of the `charge_amount`, formatted as [ISO 4127](http://www.iso.org/iso/home/standards/currency_codes.htm) format. The default value is `USD` and all values allowed from the Pay Connector are accepted. */
    currency?: string;
    /** The description can be used to provide more details regarding the transaction. This information is submitted along with the payment details to the Payment Connector which are then posted on the transactions. */
    description?: string;
    /** A list of inputs that should be accepted. Currently only `dtmf` is supported. All digits captured during a pay session are redacted from the logs. */
    input?: string;
    /** A positive integer that is used to validate the length of the `PostalCode` inputted by the user. User must enter this many digits. */
    minPostalCodeLength?: number;
    /** A single-level JSON object used to pass custom parameters to payment processors. (Required for ACH payments). The information that has to be included here depends on the <Pay> Connector. [Read more](https://www.twilio.com/console/voice/pay-connectors). */
    parameter?: any;
    /** This is the unique name corresponding to the Pay Connector installed in the Twilio Add-ons. Learn more about [<Pay> Connectors](https://www.twilio.com/console/voice/pay-connectors). The default value is `Default`. */
    paymentConnector?: string;
    /**  */
    paymentMethod?: PaymentPaymentMethod;
    /** Indicates whether the credit card postal code (zip code) is a required piece of payment information that must be provided by the caller. The default is `true`. */
    postalCode?: boolean;
    /** Indicates whether the credit card security code is a required piece of payment information that must be provided by the caller. The default is `true`. */
    securityCode?: boolean;
    /** The number of seconds that <Pay> should wait for the caller to press a digit between each subsequent digit, after the first one, before moving on to validate the digits captured. The default is `5`, maximum is `600`. */
    timeout?: number;
    /**  */
    tokenType?: PaymentTokenType;
    /** Credit card types separated by space that Pay should accept. The default value is `visa mastercard amex` */
    validCardTypes?: string;
}
export interface PaymentContext {
    /**
     * Update a PaymentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PaymentInstance
     */
    update(params: PaymentContextUpdateOptions, callback?: (error: Error | null, item?: PaymentInstance) => any): Promise<PaymentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface PaymentContextSolution {
    accountSid: string;
    callSid: string;
    sid: string;
}
export declare class PaymentContextImpl implements PaymentContext {
    protected _version: V2010;
    protected _solution: PaymentContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, callSid: string, sid: string);
    update(params: PaymentContextUpdateOptions, callback?: (error: Error | null, item?: PaymentInstance) => any): Promise<PaymentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): PaymentContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface PaymentResource {
    account_sid: string;
    call_sid: string;
    sid: string;
    date_created: Date;
    date_updated: Date;
    uri: string;
}
export declare class PaymentInstance {
    protected _version: V2010;
    protected _solution: PaymentContextSolution;
    protected _context?: PaymentContext;
    constructor(_version: V2010, payload: PaymentResource, accountSid: string, callSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Payments resource.
     */
    accountSid: string;
    /**
     * The SID of the [Call](https://www.twilio.com/docs/voice/api/call-resource) the Payments resource is associated with. This will refer to the call sid that is producing the payment card (credit/ACH) information thru DTMF.
     */
    callSid: string;
    /**
     * The SID of the Payments resource.
     */
    sid: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Update a PaymentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PaymentInstance
     */
    update(params: PaymentContextUpdateOptions, callback?: (error: Error | null, item?: PaymentInstance) => any): Promise<PaymentInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        callSid: string;
        sid: string;
        dateCreated: Date;
        dateUpdated: Date;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface PaymentSolution {
    accountSid: string;
    callSid: string;
}
export interface PaymentListInstance {
    _version: V2010;
    _solution: PaymentSolution;
    _uri: string;
    (sid: string): PaymentContext;
    get(sid: string): PaymentContext;
    /**
     * Create a PaymentInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed PaymentInstance
     */
    create(params: PaymentListInstanceCreateOptions, callback?: (error: Error | null, item?: PaymentInstance) => any): Promise<PaymentInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function PaymentListInstance(version: V2010, accountSid: string, callSid: string): PaymentListInstance;
export {};
