/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2010 from "../../V2010";
/**
 * Options to pass to create a NewKeyInstance
 */
export interface NewKeyListInstanceCreateOptions {
    /** A descriptive string that you create to describe the resource. It can be up to 64 characters long. */
    friendlyName?: string;
}
export interface NewKeySolution {
    accountSid: string;
}
export interface NewKeyListInstance {
    _version: V2010;
    _solution: NewKeySolution;
    _uri: string;
    /**
     * Create a NewKeyInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NewKeyInstance
     */
    create(callback?: (error: Error | null, item?: NewKeyInstance) => any): Promise<NewKeyInstance>;
    /**
     * Create a NewKeyInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed NewKeyInstance
     */
    create(params: NewKeyListInstanceCreateOptions, callback?: (error: Error | null, item?: NewKeyInstance) => any): Promise<NewKeyInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function NewKeyListInstance(version: V2010, accountSid: string): NewKeyListInstance;
interface NewKeyResource {
    sid: string;
    friendly_name: string;
    date_created: Date;
    date_updated: Date;
    secret: string;
}
export declare class NewKeyInstance {
    protected _version: V2010;
    constructor(_version: V2010, payload: NewKeyResource, accountSid: string);
    /**
     * The unique string that that we created to identify the NewKey resource. You will use this as the basic-auth `user` when authenticating to the API.
     */
    sid: string;
    /**
     * The string that you assigned to describe the resource.
     */
    friendlyName: string;
    /**
     * The date and time in GMT that the API Key was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the new API Key was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * The secret your application uses to sign Access Tokens and to authenticate to the REST API (you will use this as the basic-auth `password`).  **Note that for security reasons, this field is ONLY returned when the API Key is first created.**
     */
    secret: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        sid: string;
        friendlyName: string;
        dateCreated: Date;
        dateUpdated: Date;
        secret: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
