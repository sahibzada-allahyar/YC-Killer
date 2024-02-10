/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2010 from "../../../V2010";
/**
 * Options to pass to create a UserDefinedMessageInstance
 */
export interface UserDefinedMessageListInstanceCreateOptions {
    /** The User Defined Message in the form of URL-encoded JSON string. */
    content: string;
    /** A unique string value to identify API call. This should be a unique string value per API call and can be a randomly generated. */
    idempotencyKey?: string;
}
export interface UserDefinedMessageSolution {
    accountSid: string;
    callSid: string;
}
export interface UserDefinedMessageListInstance {
    _version: V2010;
    _solution: UserDefinedMessageSolution;
    _uri: string;
    /**
     * Create a UserDefinedMessageInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserDefinedMessageInstance
     */
    create(params: UserDefinedMessageListInstanceCreateOptions, callback?: (error: Error | null, item?: UserDefinedMessageInstance) => any): Promise<UserDefinedMessageInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function UserDefinedMessageListInstance(version: V2010, accountSid: string, callSid: string): UserDefinedMessageListInstance;
interface UserDefinedMessageResource {
    account_sid: string;
    call_sid: string;
    sid: string;
    date_created: Date;
}
export declare class UserDefinedMessageInstance {
    protected _version: V2010;
    constructor(_version: V2010, payload: UserDefinedMessageResource, accountSid: string, callSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created User Defined Message.
     */
    accountSid: string;
    /**
     * The SID of the [Call](https://www.twilio.com/docs/voice/api/call-resource) the User Defined Message is associated with.
     */
    callSid: string;
    /**
     * The SID that uniquely identifies this User Defined Message.
     */
    sid: string;
    /**
     * The date that this User Defined Message was created, given in RFC 2822 format.
     */
    dateCreated: Date;
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
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
