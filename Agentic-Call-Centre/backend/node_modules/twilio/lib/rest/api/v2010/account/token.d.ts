/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V2010 from "../../V2010";
export declare class ApiV2010AccountTokenIceServers {
    "credential"?: string;
    "username"?: string;
    "url"?: string;
    "urls"?: string;
}
/**
 * Options to pass to create a TokenInstance
 */
export interface TokenListInstanceCreateOptions {
    /** The duration in seconds for which the generated credentials are valid. The default value is 86400 (24 hours). */
    ttl?: number;
}
export interface TokenSolution {
    accountSid: string;
}
export interface TokenListInstance {
    _version: V2010;
    _solution: TokenSolution;
    _uri: string;
    /**
     * Create a TokenInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TokenInstance
     */
    create(callback?: (error: Error | null, item?: TokenInstance) => any): Promise<TokenInstance>;
    /**
     * Create a TokenInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed TokenInstance
     */
    create(params: TokenListInstanceCreateOptions, callback?: (error: Error | null, item?: TokenInstance) => any): Promise<TokenInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function TokenListInstance(version: V2010, accountSid: string): TokenListInstance;
interface TokenResource {
    account_sid: string;
    date_created: Date;
    date_updated: Date;
    ice_servers: Array<ApiV2010AccountTokenIceServers>;
    password: string;
    ttl: string;
    username: string;
}
export declare class TokenInstance {
    protected _version: V2010;
    constructor(_version: V2010, payload: TokenResource, accountSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Token resource.
     */
    accountSid: string;
    /**
     * The date and time in GMT that the resource was created specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateCreated: Date;
    /**
     * The date and time in GMT that the resource was last updated specified in [RFC 2822](https://www.ietf.org/rfc/rfc2822.txt) format.
     */
    dateUpdated: Date;
    /**
     * An array representing the ephemeral credentials and the STUN and TURN server URIs.
     */
    iceServers: Array<ApiV2010AccountTokenIceServers>;
    /**
     * The temporary password that the username will use when authenticating with Twilio.
     */
    password: string;
    /**
     * The duration in seconds for which the username and password are valid.
     */
    ttl: string;
    /**
     * The temporary username that uniquely identifies a Token.
     */
    username: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        dateCreated: Date;
        dateUpdated: Date;
        iceServers: ApiV2010AccountTokenIceServers[];
        password: string;
        ttl: string;
        username: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
