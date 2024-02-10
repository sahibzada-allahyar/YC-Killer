/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2010 from "../../../V2010";
/**
 * Options to pass to create a UserDefinedMessageSubscriptionInstance
 */
export interface UserDefinedMessageSubscriptionListInstanceCreateOptions {
    /** The URL we should call using the `method` to send user defined events to your application. URLs must contain a valid hostname (underscores are not permitted). */
    callback: string;
    /** A unique string value to identify API call. This should be a unique string value per API call and can be a randomly generated. */
    idempotencyKey?: string;
    /** The HTTP method Twilio will use when requesting the above `Url`. Either `GET` or `POST`. Default is `POST`. */
    method?: string;
}
export interface UserDefinedMessageSubscriptionContext {
    /**
     * Remove a UserDefinedMessageSubscriptionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface UserDefinedMessageSubscriptionContextSolution {
    accountSid: string;
    callSid: string;
    sid: string;
}
export declare class UserDefinedMessageSubscriptionContextImpl implements UserDefinedMessageSubscriptionContext {
    protected _version: V2010;
    protected _solution: UserDefinedMessageSubscriptionContextSolution;
    protected _uri: string;
    constructor(_version: V2010, accountSid: string, callSid: string, sid: string);
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): UserDefinedMessageSubscriptionContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface UserDefinedMessageSubscriptionResource {
    account_sid: string;
    call_sid: string;
    sid: string;
    date_created: Date;
    uri: string;
}
export declare class UserDefinedMessageSubscriptionInstance {
    protected _version: V2010;
    protected _solution: UserDefinedMessageSubscriptionContextSolution;
    protected _context?: UserDefinedMessageSubscriptionContext;
    constructor(_version: V2010, payload: UserDefinedMessageSubscriptionResource, accountSid: string, callSid: string, sid?: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that subscribed to the User Defined Messages.
     */
    accountSid: string;
    /**
     * The SID of the [Call](https://www.twilio.com/docs/voice/api/call-resource) the User Defined Message Subscription is associated with. This refers to the Call SID that is producing the User Defined Messages.
     */
    callSid: string;
    /**
     * The SID that uniquely identifies this User Defined Message Subscription.
     */
    sid: string;
    /**
     * The date that this User Defined Message Subscription was created, given in RFC 2822 format.
     */
    dateCreated: Date;
    /**
     * The URI of the User Defined Message Subscription Resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    private get _proxy();
    /**
     * Remove a UserDefinedMessageSubscriptionInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed boolean
     */
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
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
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface UserDefinedMessageSubscriptionSolution {
    accountSid: string;
    callSid: string;
}
export interface UserDefinedMessageSubscriptionListInstance {
    _version: V2010;
    _solution: UserDefinedMessageSubscriptionSolution;
    _uri: string;
    (sid: string): UserDefinedMessageSubscriptionContext;
    get(sid: string): UserDefinedMessageSubscriptionContext;
    /**
     * Create a UserDefinedMessageSubscriptionInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed UserDefinedMessageSubscriptionInstance
     */
    create(params: UserDefinedMessageSubscriptionListInstanceCreateOptions, callback?: (error: Error | null, item?: UserDefinedMessageSubscriptionInstance) => any): Promise<UserDefinedMessageSubscriptionInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function UserDefinedMessageSubscriptionListInstance(version: V2010, accountSid: string, callSid: string): UserDefinedMessageSubscriptionListInstance;
export {};
