/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2 from "../../../V2";
export type ReplaceItemsStatus = "draft" | "pending-review" | "in-review" | "twilio-rejected" | "twilio-approved" | "provisionally-approved";
/**
 * Options to pass to create a ReplaceItemsInstance
 */
export interface ReplaceItemsListInstanceCreateOptions {
    /** The source bundle sid to copy the item assignments from. */
    fromBundleSid: string;
}
export interface ReplaceItemsSolution {
    bundleSid: string;
}
export interface ReplaceItemsListInstance {
    _version: V2;
    _solution: ReplaceItemsSolution;
    _uri: string;
    /**
     * Create a ReplaceItemsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed ReplaceItemsInstance
     */
    create(params: ReplaceItemsListInstanceCreateOptions, callback?: (error: Error | null, item?: ReplaceItemsInstance) => any): Promise<ReplaceItemsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function ReplaceItemsListInstance(version: V2, bundleSid: string): ReplaceItemsListInstance;
interface ReplaceItemsResource {
    sid: string;
    account_sid: string;
    regulation_sid: string;
    friendly_name: string;
    status: ReplaceItemsStatus;
    valid_until: Date;
    email: string;
    status_callback: string;
    date_created: Date;
    date_updated: Date;
}
export declare class ReplaceItemsInstance {
    protected _version: V2;
    constructor(_version: V2, payload: ReplaceItemsResource, bundleSid: string);
    /**
     * The unique string that we created to identify the Bundle resource.
     */
    sid: string;
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Bundle resource.
     */
    accountSid: string;
    /**
     * The unique string of a regulation that is associated to the Bundle resource.
     */
    regulationSid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    status: ReplaceItemsStatus;
    /**
     * The date and time in GMT in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format when the resource will be valid until.
     */
    validUntil: Date;
    /**
     * The email address that will receive updates when the Bundle resource changes status.
     */
    email: string;
    /**
     * The URL we call to inform your application of status changes.
     */
    statusCallback: string;
    /**
     * The date and time in GMT when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        accountSid: string;
        regulationSid: string;
        friendlyName: string;
        status: ReplaceItemsStatus;
        validUntil: Date;
        email: string;
        statusCallback: string;
        dateCreated: Date;
        dateUpdated: Date;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
