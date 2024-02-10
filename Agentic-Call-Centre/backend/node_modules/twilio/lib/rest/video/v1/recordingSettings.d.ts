/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../V1";
/**
 * Options to pass to create a RecordingSettingsInstance
 */
export interface RecordingSettingsContextCreateOptions {
    /** A descriptive string that you create to describe the resource and be shown to users in the console */
    friendlyName: string;
    /** The SID of the stored Credential resource. */
    awsCredentialsSid?: string;
    /** The SID of the Public Key resource to use for encryption. */
    encryptionKeySid?: string;
    /** The URL of the AWS S3 bucket where the recordings should be stored. We only support DNS-compliant URLs like `https://documentation-example-twilio-bucket/recordings`, where `recordings` is the path in which you want the recordings to be stored. This URL accepts only URI-valid characters, as described in the [RFC 3986](https://tools.ietf.org/html/rfc3986#section-2). */
    awsS3Url?: string;
    /** Whether all recordings should be written to the `aws_s3_url`. When `false`, all recordings are stored in our cloud. */
    awsStorageEnabled?: boolean;
    /** Whether all recordings should be stored in an encrypted form. The default is `false`. */
    encryptionEnabled?: boolean;
}
export interface RecordingSettingsContext {
    /**
     * Create a RecordingSettingsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingSettingsInstance
     */
    create(params: RecordingSettingsContextCreateOptions, callback?: (error: Error | null, item?: RecordingSettingsInstance) => any): Promise<RecordingSettingsInstance>;
    /**
     * Fetch a RecordingSettingsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingSettingsInstance
     */
    fetch(callback?: (error: Error | null, item?: RecordingSettingsInstance) => any): Promise<RecordingSettingsInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export interface RecordingSettingsContextSolution {
}
export declare class RecordingSettingsContextImpl implements RecordingSettingsContext {
    protected _version: V1;
    protected _solution: RecordingSettingsContextSolution;
    protected _uri: string;
    constructor(_version: V1);
    create(params: RecordingSettingsContextCreateOptions, callback?: (error: Error | null, item?: RecordingSettingsInstance) => any): Promise<RecordingSettingsInstance>;
    fetch(callback?: (error: Error | null, item?: RecordingSettingsInstance) => any): Promise<RecordingSettingsInstance>;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): RecordingSettingsContextSolution;
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
interface RecordingSettingsResource {
    account_sid: string;
    friendly_name: string;
    aws_credentials_sid: string;
    aws_s3_url: string;
    aws_storage_enabled: boolean;
    encryption_key_sid: string;
    encryption_enabled: boolean;
    url: string;
}
export declare class RecordingSettingsInstance {
    protected _version: V1;
    protected _solution: RecordingSettingsContextSolution;
    protected _context?: RecordingSettingsContext;
    constructor(_version: V1, payload: RecordingSettingsResource);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the RecordingSettings resource.
     */
    accountSid: string;
    /**
     * The string that you assigned to describe the resource and show the user in the console
     */
    friendlyName: string;
    /**
     * The SID of the stored Credential resource.
     */
    awsCredentialsSid: string;
    /**
     * The URL of the AWS S3 bucket where the recordings are stored. We only support DNS-compliant URLs like `https://documentation-example-twilio-bucket/recordings`, where `recordings` is the path in which you want the recordings to be stored. This URL accepts only URI-valid characters, as described in the [RFC 3986](https://tools.ietf.org/html/rfc3986#section-2).
     */
    awsS3Url: string;
    /**
     * Whether all recordings are written to the `aws_s3_url`. When `false`, all recordings are stored in our cloud.
     */
    awsStorageEnabled: boolean;
    /**
     * The SID of the Public Key resource used for encryption.
     */
    encryptionKeySid: string;
    /**
     * Whether all recordings are stored in an encrypted form. The default is `false`.
     */
    encryptionEnabled: boolean;
    /**
     * The absolute URL of the resource.
     */
    url: string;
    private get _proxy();
    /**
     * Create a RecordingSettingsInstance
     *
     * @param params - Parameter for request
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingSettingsInstance
     */
    create(params: RecordingSettingsContextCreateOptions, callback?: (error: Error | null, item?: RecordingSettingsInstance) => any): Promise<RecordingSettingsInstance>;
    /**
     * Fetch a RecordingSettingsInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed RecordingSettingsInstance
     */
    fetch(callback?: (error: Error | null, item?: RecordingSettingsInstance) => any): Promise<RecordingSettingsInstance>;
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
export interface RecordingSettingsSolution {
}
export interface RecordingSettingsListInstance {
    _version: V1;
    _solution: RecordingSettingsSolution;
    _uri: string;
    (): RecordingSettingsContext;
    get(): RecordingSettingsContext;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function RecordingSettingsListInstance(version: V1): RecordingSettingsListInstance;
export {};
