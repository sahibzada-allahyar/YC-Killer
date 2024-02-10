/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to create a CompositionSettingsInstance
 */
export interface CompositionSettingsContextCreateOptions {
    /** A descriptive string that you create to describe the resource and show to the user in the console */
    friendlyName: string;
    /** The SID of the stored Credential resource. */
    awsCredentialsSid?: string;
    /** The SID of the Public Key resource to use for encryption. */
    encryptionKeySid?: string;
    /** The URL of the AWS S3 bucket where the compositions should be stored. We only support DNS-compliant URLs like `https://documentation-example-twilio-bucket/compositions`, where `compositions` is the path in which you want the compositions to be stored. This URL accepts only URI-valid characters, as described in the [RFC 3986](https://tools.ietf.org/html/rfc3986#section-2). */
    awsS3Url?: string;
    /** Whether all compositions should be written to the `aws_s3_url`. When `false`, all compositions are stored in our cloud. */
    awsStorageEnabled?: boolean;
    /** Whether all compositions should be stored in an encrypted form. The default is `false`. */
    encryptionEnabled?: boolean;
}
export interface CompositionSettingsContext {
    /**
     * Create a CompositionSettingsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CompositionSettingsInstance
     */
    create(params: CompositionSettingsContextCreateOptions, callback?: (error: Error | null, item?: CompositionSettingsInstance) => any): Promise<CompositionSettingsInstance>;
    /**
     * Fetch a CompositionSettingsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CompositionSettingsInstance
     */
    fetch(callback?: (error: Error | null, item?: CompositionSettingsInstance) => any): Promise<CompositionSettingsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface CompositionSettingsContextSolution {
}
export declare class CompositionSettingsContextImpl implements CompositionSettingsContext {
    protected _version: V1;
    protected _solution: CompositionSettingsContextSolution;
    protected _uri: string;
    constructor(_version: V1);
    create(params: CompositionSettingsContextCreateOptions, callback?: (error: Error | null, item?: CompositionSettingsInstance) => any): Promise<CompositionSettingsInstance>;
    fetch(callback?: (error: Error | null, item?: CompositionSettingsInstance) => any): Promise<CompositionSettingsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): CompositionSettingsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface CompositionSettingsResource {
    account_sid: string;
    friendly_name: string;
    aws_credentials_sid: string;
    aws_s3_url: string;
    aws_storage_enabled: boolean;
    encryption_key_sid: string;
    encryption_enabled: boolean;
    url: string;
}
export declare class CompositionSettingsInstance {
    protected _version: V1;
    protected _solution: CompositionSettingsContextSolution;
    protected _context?: CompositionSettingsContext;
    constructor(_version: V1, payload: CompositionSettingsResource);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the CompositionSettings resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the resource and that will be shown in the console
     */
    friendlyName: string;
    /**
     * The SID of the stored Credential resource.
     */
    awsCredentialsSid: string;
    /**
     * The URL of the AWS S3 bucket where the compositions are stored. We only support DNS-compliant URLs like `https://documentation-example-twilio-bucket/compositions`, where `compositions` is the path in which you want the compositions to be stored. This URL accepts only URI-valid characters, as described in the [RFC 3986](https://tools.ietf.org/html/rfc3986#section-2).
     */
    awsS3Url: string;
    /**
     * Whether all compositions are written to the `aws_s3_url`. When `false`, all compositions are stored in our cloud.
     */
    awsStorageEnabled: boolean;
    /**
     * The SID of the Public Key resource used for encryption.
     */
    encryptionKeySid: string;
    /**
     * Whether all compositions are stored in an encrypted form. The default is `false`.
     */
    encryptionEnabled: boolean;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Create a CompositionSettingsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CompositionSettingsInstance
     */
    create(params: CompositionSettingsContextCreateOptions, callback?: (error: Error | null, item?: CompositionSettingsInstance) => any): Promise<CompositionSettingsInstance>;
    /**
     * Fetch a CompositionSettingsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed CompositionSettingsInstance
     */
    fetch(callback?: (error: Error | null, item?: CompositionSettingsInstance) => any): Promise<CompositionSettingsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        friendlyName: string;
        awsCredentialsSid: string;
        awsS3Url: string;
        awsStorageEnabled: boolean;
        encryptionKeySid: string;
        encryptionEnabled: boolean;
        url: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export interface CompositionSettingsSolution {
}
export interface CompositionSettingsListInstance {
    _version: V1;
    _solution: CompositionSettingsSolution;
    _uri: string;
    (): CompositionSettingsContext;
    get(): CompositionSettingsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function CompositionSettingsListInstance(version: V1): CompositionSettingsListInstance;
export {};
