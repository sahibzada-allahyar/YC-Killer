/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2010 from "../../V2010";
/**
 * Options to pass to create a ValidationRequestInstance
 */
export interface ValidationRequestListInstanceCreateOptions {
    /** The phone number to verify in [E.164](https://www.twilio.com/docs/glossary/what-e164) format, which consists of a + followed by the country code and subscriber number. */
    phoneNumber: string;
    /** A descriptive string that you create to describe the new caller ID resource. It can be up to 64 characters long. The default value is a formatted version of the phone number. */
    friendlyName?: string;
    /** The number of seconds to delay before initiating the verification call. Can be an integer between `0` and `60`, inclusive. The default is `0`. */
    callDelay?: number;
    /** The digits to dial after connecting the verification call. */
    extension?: string;
    /** The URL we should call using the `status_callback_method` to send status information about the verification process to your application. */
    statusCallback?: string;
    /** The HTTP method we should use to call `status_callback`. Can be: `GET` or `POST`, and the default is `POST`. */
    statusCallbackMethod?: string;
}
export interface ValidationRequestSolution {
    accountSid: string;
}
export interface ValidationRequestListInstance {
    _version: V2010;
    _solution: ValidationRequestSolution;
    _uri: string;
    /**
     * Create a ValidationRequestInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ValidationRequestInstance
     */
    create(params: ValidationRequestListInstanceCreateOptions, callback?: (error: Error | null, item?: ValidationRequestInstance) => any): Promise<ValidationRequestInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ValidationRequestListInstance(version: V2010, accountSid: string): ValidationRequestListInstance;
interface ValidationRequestResource {
    account_sid: string;
    call_sid: string;
    friendly_name: string;
    phone_number: string;
    validation_code: string;
}
export declare class ValidationRequestInstance {
    protected _version: V2010;
    constructor(_version: V2010, payload: ValidationRequestResource, accountSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) responsible for the Caller ID.
     */
    accountSid: string;
    /**
     * The SID of the [Call](https://www.twilio.com/docs/voice/api/call-resource) the Caller ID is associated with.
     */
    callSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The phone number to verify in [E.164](https://www.twilio.com/docs/glossary/what-e164) format, which consists of a + followed by the country code and subscriber number.
     */
    phoneNumber: string;
    /**
     * The 6 digit validation code that someone must enter to validate the Caller ID  when `phone_number` is called.
     */
    validationCode: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        callSid: string;
        friendlyName: string;
        phoneNumber: string;
        validationCode: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
