/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2010 from "../../../V2010";
export type FeedbackOutcome = "confirmed" | "unconfirmed";
/**
 * Options to pass to create a FeedbackInstance
 */
export interface FeedbackListInstanceCreateOptions {
    /**  */
    outcome?: FeedbackOutcome;
}
export interface FeedbackSolution {
    accountSid: string;
    messageSid: string;
}
export interface FeedbackListInstance {
    _version: V2010;
    _solution: FeedbackSolution;
    _uri: string;
    /**
     * Create a FeedbackInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FeedbackInstance
     */
    create(callback?: (error: Error | null, item?: FeedbackInstance) => any): Promise<FeedbackInstance>;
    /**
     * Create a FeedbackInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed FeedbackInstance
     */
    create(params: FeedbackListInstanceCreateOptions, callback?: (error: Error | null, item?: FeedbackInstance) => any): Promise<FeedbackInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function FeedbackListInstance(version: V2010, accountSid: string, messageSid: string): FeedbackListInstance;
interface FeedbackResource {
    account_sid: string;
    message_sid: string;
    outcome: FeedbackOutcome;
    date_created: Date;
    date_updated: Date;
    uri: string;
}
export declare class FeedbackInstance {
    protected _version: V2010;
    constructor(_version: V2010, payload: FeedbackResource, accountSid: string, messageSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) associated with this MessageFeedback resource.
     */
    accountSid: string;
    /**
     * The SID of the Message resource associated with this MessageFeedback resource.
     */
    messageSid: string;
    outcome: FeedbackOutcome;
    /**
     * The date and time in GMT when this MessageFeedback resource was created, specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when this MessageFeedback resource was last updated, specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The URI of the resource, relative to `https://api.twilio.com`.
     */
    uri: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        messageSid: string;
        outcome: FeedbackOutcome;
        dateCreated: Date;
        dateUpdated: Date;
        uri: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
