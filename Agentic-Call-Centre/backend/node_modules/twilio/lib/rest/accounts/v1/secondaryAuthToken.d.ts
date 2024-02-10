/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
export interface SecondaryAuthTokenContext {
    /**
     * Create a SecondaryAuthTokenInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SecondaryAuthTokenInstance
     */
    create(callback?: (error: Error | null, item?: SecondaryAuthTokenInstance) => any): Promise<SecondaryAuthTokenInstance>;
    /**
     * Remove a SecondaryAuthTokenInstance
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
export interface SecondaryAuthTokenContextSolution {
}
export declare class SecondaryAuthTokenContextImpl implements SecondaryAuthTokenContext {
    protected _version: V1;
    protected _solution: SecondaryAuthTokenContextSolution;
    protected _uri: string;
    constructor(_version: V1);
    create(callback?: (error: Error | null, item?: SecondaryAuthTokenInstance) => any): Promise<SecondaryAuthTokenInstance>;
    remove(callback?: (error: Error | null, item?: boolean) => any): Promise<boolean>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): SecondaryAuthTokenContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface SecondaryAuthTokenResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    secondary_auth_token: string;
    url: string;
}
export declare class SecondaryAuthTokenInstance {
    protected _version: V1;
    protected _solution: SecondaryAuthTokenContextSolution;
    protected _context?: SecondaryAuthTokenContext;
    constructor(_version: V1, payload: SecondaryAuthTokenResource);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that the secondary Auth Token was created for.
     */
    accountSid: string;
    /**
     * The date and time in UTC when the resource was created specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateCreated: Date;
    /**
     * The date and time in UTC when the resource was last updated specified in [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) format.
     */
    dateUpdated: Date;
    /**
     * The generated secondary Auth Token that can be used to authenticate future API requests.
     */
    secondaryAuthToken: string;
    /**
     * The URI for this resource, relative to `https://accounts.twilio.com`
     */
    url: string;
    private get _proxy();
    /**
     * Create a SecondaryAuthTokenInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed SecondaryAuthTokenInstance
     */
    create(callback?: (error: Error | null, item?: SecondaryAuthTokenInstance) => any): Promise<SecondaryAuthTokenInstance>;
    /**
     * Remove a SecondaryAuthTokenInstance
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
        dateCreated: Date;
        dateUpdated: Date;
        secondaryAuthToken: string;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface SecondaryAuthTokenSolution {
}
export interface SecondaryAuthTokenListInstance {
    _version: V1;
    _solution: SecondaryAuthTokenSolution;
    _uri: string;
    (): SecondaryAuthTokenContext;
    get(): SecondaryAuthTokenContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function SecondaryAuthTokenListInstance(version: V1): SecondaryAuthTokenListInstance;
export {};
