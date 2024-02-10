/// <reference types="node" />
import { inspect, InspectOptions } from "util";
import V1 from "../../V1";
export interface BrandRegistrationOtpSolution {
    brandRegistrationSid: string;
}
export interface BrandRegistrationOtpListInstance {
    _version: V1;
    _solution: BrandRegistrationOtpSolution;
    _uri: string;
    /**
     * Create a BrandRegistrationOtpInstance
     *
     * @param callback - Callback to handle processed record
     *
     * @returns Resolves to processed BrandRegistrationOtpInstance
     */
    create(callback?: (error: Error | null, item?: BrandRegistrationOtpInstance) => any): Promise<BrandRegistrationOtpInstance>;
    /**
     * Provide a user-friendly representation
     */
    toJSON(): any;
    [inspect.custom](_depth: any, options: InspectOptions): any;
}
export declare function BrandRegistrationOtpListInstance(version: V1, brandRegistrationSid: string): BrandRegistrationOtpListInstance;
interface BrandRegistrationOtpResource {
    account_sid: string;
    brand_registration_sid: string;
}
export declare class BrandRegistrationOtpInstance {
    protected _version: V1;
    constructor(_version: V1, payload: BrandRegistrationOtpResource, brandRegistrationSid: string);
    /**
     * The SID of the [Account](https://www.twilio.com/docs/iam/api/account) that created the Brand Registration resource.
     */
    accountSid: string;
    /**
     * The unique string to identify Brand Registration of Sole Proprietor Brand
     */
    brandRegistrationSid: string;
    /**
     * Provide a user-friendly representation
     *
     * @returns Object
     */
    toJSON(): {
        accountSid: string;
        brandRegistrationSid: string;
    };
    [inspect.custom](_depth: any, options: InspectOptions): string;
}
export {};
